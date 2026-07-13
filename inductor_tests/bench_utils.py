import glob
import hashlib
import os
import shutil
import subprocess
import time
import uuid

from datetime import datetime, timezone
from typing import Callable, Dict

import torch


def _is_npu_device(device=None) -> bool:
    if device is None:
        return hasattr(torch, "npu") and torch.npu.is_available()
    device_type = device.type if isinstance(device, torch.device) else str(device)
    return device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available()


def synchronize(device=None) -> None:
    if device is None:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        return

    device_type = device.type if isinstance(device, torch.device) else str(device)
    if device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


class SimpleProfilingAnalyzer:
    def __init__(self, output_dir: str = None, worker_name: str = None):
        from torch_npu.profiler._profiler_path_creator import ProfPathCreator

        self.output_dir = output_dir
        self.op_summary_files = []
        self.op_statistic_files = []
        self.use_custom_analyzer = True
        ProfPathCreator().init(dir_name=output_dir, worker_name=worker_name)

    def _get_msprof_py_script_path(self):
        from torch_npu._inductor.cpp_builder import get_ascend_home

        ascend_home_path = get_ascend_home()
        script_path = "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
        full_script_path = os.path.join(ascend_home_path, script_path)
        return full_script_path if os.path.exists(full_script_path) else None

    def _msprof_py_export(self, msprof_py_script_path, ascend_pt_dir) -> None:
        if not (os.path.exists(msprof_py_script_path) and os.path.exists(ascend_pt_dir)):
            raise RuntimeError("Failed to run export subprocess, command or ascend_pt dir not found.")
        prof_dir = None
        for item in os.listdir(ascend_pt_dir):
            item_path = os.path.join(ascend_pt_dir, item)
            if os.path.isdir(item_path) and item.startswith("PROF"):
                prof_dir = item_path
                break
        if prof_dir is None:
            raise RuntimeError("PROF directory not found.")
        mindstudio_profiler_output_dir = os.path.join(prof_dir, "mindstudio_profiler_output")
        if os.path.exists(mindstudio_profiler_output_dir):
            shutil.rmtree(mindstudio_profiler_output_dir)
        export_cmd = ["python", msprof_py_script_path, "export", "summary", "-dir", prof_dir]
        completed_analysis = subprocess.run(export_cmd, capture_output=True)
        if completed_analysis.returncode != 0:
            raise RuntimeError("subprocess return code is not 0.")
        self.op_summary_files = glob.glob(os.path.join(mindstudio_profiler_output_dir, "op_summary*.csv"))
        self.op_statistic_files = glob.glob(os.path.join(mindstudio_profiler_output_dir, "op_statistic*.csv"))
        if not self.op_summary_files:
            raise RuntimeError("export results not found.")

    def _convert_op_summary_to_kernel_details(self):
        if not self.op_summary_files:
            return
        ascend_profiler_output_dir = os.path.join(self.ascend_pt_dir, "ASCEND_PROFILER_OUTPUT")
        kernel_details_path = os.path.join(ascend_profiler_output_dir, "kernel_details.csv")
        os.makedirs(ascend_profiler_output_dir, exist_ok=True)
        if len(self.op_summary_files) == 1:
            header_replace_mapping = {",Op Name,": ",Name,", ",Task Duration(us),": ",Duration(us),"}
            with open(self.op_summary_files[0], "r", encoding="utf-8") as op_summary_file:
                lines = op_summary_file.readlines()
            if not lines:
                raise RuntimeError("convert op_summary to kernel_details failed, op_summary file empty.")
            original_header = lines[0]
            for source, target in header_replace_mapping.items():
                lines[0] = lines[0].replace(source, target)
            if lines[0] == original_header:
                raise RuntimeError("convert op_summary to kernel_details failed, replace header failed.")
            with open(kernel_details_path, "w", encoding="utf-8") as kernel_details_file:
                kernel_details_file.writelines(lines)
            return

        import pandas as pd

        merged_df = pd.concat((pd.read_csv(file) for file in self.op_summary_files), ignore_index=True)
        merged_df = merged_df.sort_values(by="Task ID")
        merged_df.rename(columns={"Op Name": "Name", "Task Duration(us)": "Duration(us)"}, inplace=True)
        merged_df.to_csv(kernel_details_path, index=False, encoding="utf-8")

    def _convert_op_statistic(self):
        if not self.op_statistic_files:
            return
        ascend_profiler_output_dir = os.path.join(self.ascend_pt_dir, "ASCEND_PROFILER_OUTPUT")
        op_statistic_path = os.path.join(ascend_profiler_output_dir, "op_statistic.csv")
        os.makedirs(ascend_profiler_output_dir, exist_ok=True)
        if len(self.op_statistic_files) == 1:
            shutil.copy(self.op_statistic_files[0], op_statistic_path)
            return

        import pandas as pd

        merged_df = pd.concat((pd.read_csv(file) for file in self.op_statistic_files), ignore_index=True)
        merged_df.to_csv(op_statistic_path, index=False, encoding="utf-8")

    def trace_ready(self, prof_inst):
        msprof_py_script_path = self._get_msprof_py_script_path()
        self.ascend_pt_dir = prof_inst.prof_if.prof_path
        try:
            print(f"[bench_utils] Start parsing profiling data: {self.ascend_pt_dir}")
            export_start = datetime.now(tz=timezone.utc).astimezone()
            self._msprof_py_export(msprof_py_script_path, self.ascend_pt_dir)
            self._convert_op_summary_to_kernel_details()
            self._convert_op_statistic()
            export_end = datetime.now(tz=timezone.utc).astimezone()
            print(f"[bench_utils] Profiling data parsed in {export_end - export_start}")
        except Exception as exc:
            print(f"[bench_utils] Failed to parse profiling data: {exc}. Fallback to default analysis.")
            self.use_custom_analyzer = False
            prof_inst.prof_if.analyse(async_mode=False)


def simple_trace_handler(dir_name: str = None, worker_name: str = None):
    analyzer = SimpleProfilingAnalyzer(dir_name, worker_name)

    def trace_ready(prof_inst):
        try:
            analyzer.trace_ready(prof_inst)
        except Exception as exc:
            print(f"[bench_utils] Trace handler failed: {exc}")

    return trace_ready


def _filter_rows(df, filter_list):
    if not filter_list:
        return df

    import pandas as pd

    mask = pd.Series([False] * len(df), index=df.index)
    for pattern in filter_list:
        mask |= df["Name"].str.contains(pattern, case=False, na=False)
    return df[mask]


def _delete_file(base_path):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)


def _profile_npu(fn: Callable, warmup=5, active=30, filter_list=None) -> Dict[str, float]:
    import pandas as pd
    import torch_npu

    fn()
    torch.npu.synchronize()

    random_uuid = uuid.uuid4().hex
    md5_hash = hashlib.md5(random_uuid.encode()).hexdigest()
    torch_path = os.path.join(os.getcwd(), "profile_result", f"triton_{md5_hash}")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        l2_cache=False,
        data_simplification=False,
    )

    wait = 1
    repeat = 1
    skip_first = 1
    total_step = (wait + warmup + active + skip_first) * repeat
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            skip_first=skip_first,
        ),
        on_trace_ready=simple_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total_step):
            fn()
            torch.npu.synchronize()
            prof.step()

    for root, _, files in os.walk(torch_path):
        for file in files:
            if file != "kernel_details.csv":
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            filtered_df = _filter_rows(df, filter_list)
            time_cost_df = filtered_df.groupby("Name")["Duration(us)"].mean().to_dict()
            time_cost_df["__total_us__"] = float(filtered_df["Duration(us)"].sum() / max(active, 1))
            time_cost_df["__backend__"] = "npu_profile"
            _delete_file(torch_path)
            return time_cost_df

    _delete_file(torch_path)
    return {"__total_us__": 0.0, "__backend__": "npu_profile"}


def _profile_fallback(fn: Callable, warmup=5, active=30, device=None) -> Dict[str, float]:
    for _ in range(max(warmup, 0)):
        fn()
    synchronize(device)

    started = time.perf_counter()
    for _ in range(max(active, 1)):
        fn()
    synchronize(device)
    elapsed_us = (time.perf_counter() - started) * 1e6 / max(active, 1)
    return {
        "__total_us__": float(elapsed_us),
        "__backend__": "wall_time",
    }


def profile(fn: Callable, warmup=5, active=30, filter_list=None, device=None) -> Dict[str, float]:
    if _is_npu_device(device):
        try:
            return _profile_npu(fn, warmup=warmup, active=active, filter_list=filter_list)
        except Exception as exc:
            print(f"[bench_utils] NPU profiling failed, fallback to wall time: {exc}")
    return _profile_fallback(fn, warmup=warmup, active=active, device=device)
