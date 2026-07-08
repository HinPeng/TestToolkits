import glob
import hashlib
import os
import shutil
import subprocess
import uuid

from datetime import datetime, timezone
from typing import Callable, Dict

from torch_npu.profiler.analysis.prof_common_func._constant import print_warn_msg, print_info_msg
from torch_npu.profiler.analysis.prof_common_func._utils import no_exception_func
from torch_npu.profiler._profiler_path_creator import ProfPathCreator


class SimpleProfilingAnalyzer:
    def __init__(self, output_dir: str = None, worker_name: str = None):
        self.output_dir = output_dir
        ProfPathCreator().init(dir_name=output_dir, worker_name=worker_name)
        self.op_summary_files = None
        self.op_statistic_files = None
        self.use_custom_analyzer = True

    def _get_msprof_py_script_path(self):
        from torch_npu._inductor.cpp_builder import get_ascend_home

        ascend_home_path = get_ascend_home()
        script_path = "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
        full_script_path = os.path.join(ascend_home_path, script_path)
        if os.path.exists(full_script_path):
            return full_script_path
        else:
            return None

    def _msprof_py_export(self, msprof_py_script_path, ascend_pt_dir) -> None:
        if not (os.path.exists(msprof_py_script_path) and os.path.exists(ascend_pt_dir)):
            raise RuntimeError("Failed to run export subprocess, command or ascend_pt dir not found.")
        prof_dir = None
        for item in os.listdir(ascend_pt_dir):
            item_path = os.path.join(ascend_pt_dir, item)
            if os.path.isdir(item_path) and item.startswith('PROF'):
                prof_dir = item_path
                break
        if prof_dir is None:
            raise RuntimeError("PROF directory not found.")
        mindstudio_profiler_output_dir = os.path.join(prof_dir, "mindstudio_profiler_output")
        if os.path.exists(mindstudio_profiler_output_dir):
            shutil.rmtree(mindstudio_profiler_output_dir)
        export_cmd = ['python', msprof_py_script_path, "export", "summary", "-dir", prof_dir]
        completed_analysis = subprocess.run(export_cmd, capture_output=True)
        if completed_analysis.returncode != 0:
            raise RuntimeError("subprocess return code is not 0.")
        op_summary_file_pattern = os.path.join(mindstudio_profiler_output_dir, "op_summary*.csv")
        op_summary_files = glob.glob(op_summary_file_pattern)
        self.op_summary_files = op_summary_files
        op_statistic_file_pattern = os.path.join(mindstudio_profiler_output_dir, "op_statistic*.csv")
        op_statistic_files = glob.glob(op_statistic_file_pattern)
        self.op_statistic_files = op_statistic_files
        if len(op_summary_files) == 0:
            raise RuntimeError("export results not found.")

    def _convert_op_summary_to_kernel_details(self):
        if len(self.op_summary_files) == 0:
            return
        ascend_profiler_output_dir = os.path.join(self.ascend_pt_dir, "ASCEND_PROFILER_OUTPUT")
        kernel_details_path = os.path.join(ascend_profiler_output_dir, "kernel_details.csv")
        os.makedirs(ascend_profiler_output_dir, exist_ok=True)
        if len(self.op_summary_files) == 1:
            header_replace_mapping = {',Op Name,': ',Name,', ',Task Duration(us),': ',Duration(us),'}
            with open(self.op_summary_files[0], 'r', encoding='utf-8') as op_summary_file:
                lines = op_summary_file.readlines()
                if lines:
                    ori_line0 = lines[0]
                    for ori_str, sub_str in header_replace_mapping.items():
                        lines[0] = lines[0].replace(ori_str, sub_str)
                    if lines[0] == ori_line0:
                        raise RuntimeError("convert op_summary to kernel_details failed, replace header failed.")
                    with open(kernel_details_path, 'w', encoding='utf-8') as kernel_details_file:
                        kernel_details_file.writelines(lines)
                else:
                    raise RuntimeError("convert op_summary to kernel_details failed, op_summary file empty.")
        else:
            import pandas as pd
            df_list = []
            for file in self.op_summary_files:
                df = pd.read_csv(file)
                df_list.append(df)

            merged_df = pd.concat(df_list, ignore_index=True)
            merged_df = merged_df.sort_values(by='Task ID')
            column_rename_mapping = {'Op Name': 'Name', 'Task Duration(us)': 'Duration(us)'}
            merged_df.rename(columns=column_rename_mapping, inplace=True)
            merged_df.to_csv(kernel_details_path, index=False, encoding="utf-8")

    def _convert_op_statistic(self):
        if len(self.op_statistic_files) == 0:
            return
        ascend_profiler_output_dir = os.path.join(self.ascend_pt_dir, "ASCEND_PROFILER_OUTPUT")
        op_statistic_path = os.path.join(ascend_profiler_output_dir, "op_statistic.csv")
        os.makedirs(ascend_profiler_output_dir, exist_ok=True)
        if len(self.op_statistic_files) == 1:
            shutil.copy(self.op_statistic_files[0], op_statistic_path)
        else:
            import pandas as pd
            df_list = []
            for file in self.op_statistic_files:
                df = pd.read_csv(file)
                df_list.append(df)
            merged_df = pd.concat(df_list, ignore_index=True)
            merged_df.to_csv(op_statistic_path, index=False, encoding="utf-8")

    def trace_ready(self, prof_inst):
        msprof_py_script_path = self._get_msprof_py_script_path()
        ascend_pt_dir = prof_inst.prof_if.prof_path
        self.ascend_pt_dir = ascend_pt_dir
        try:
            print_info_msg(f'Start parsing profiling data: {ascend_pt_dir}')
            export_start = datetime.now(tz=timezone.utc).astimezone()
            self._msprof_py_export(msprof_py_script_path, ascend_pt_dir)
            self._convert_op_summary_to_kernel_details()
            self._convert_op_statistic()
            export_end = datetime.now(tz=timezone.utc).astimezone()
            print_info_msg(f'SimpleProfilingAnalyzer parsed in a total time of {export_end - export_start}')
        except Exception as e:
            # fullback to default tensorboard_trace_handler
            print_warn_msg(f'Failed to run msprof.py subprocess, reason: {e}, fallback to tensorboard_trace_handler.')
            self.use_custom_analyzer = False
            prof_inst.prof_if.analyse(async_mode=False)


@no_exception_func()
def simple_trace_handler(dir_name: str = None, worker_name: str = None):
    prof_analyzer = SimpleProfilingAnalyzer(dir_name, worker_name)
    return prof_analyzer.trace_ready


"""
Extract kernels execution time
"""
def profile(fn: Callable, warmup=5, active=30, filter_list=None) -> Dict[str, float]:
    import pandas as pd
    import torch
    import torch_npu

    def filter_rows(df, filter_list):
        if not filter_list:
            return df
        mask = pd.Series([False] * len(df), index=df.index)
        for pattern in filter_list:
            mask |= df['Name'].str.contains(pattern, case=False, na=False)
        return df[mask]

    def delete_file(base_path):
        if os.path.exists(base_path):
            shutil.rmtree(base_path)

    # warmup
    fn()
    torch.npu.synchronize()

    random_uuid = uuid.uuid4().hex
    md5_hash = hashlib.md5(random_uuid.encode()).hexdigest()
    torch_path = os.path.join(os.getcwd(), "profile_result", f"triton_{md5_hash}")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        l2_cache=False,
        data_simplification=False
    )

    WAIT = 1
    WARMUP = warmup
    ACTIVE = active
    REPEAT = 1
    SKIP_FIRST = 1
    TOTAL_STEP = (WAIT + WARMUP + ACTIVE + SKIP_FIRST) * REPEAT
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=REPEAT, skip_first=SKIP_FIRST),
        on_trace_ready=simple_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config
    ) as prof:
        for _ in range(TOTAL_STEP):
            fn()
            torch.npu.synchronize()
            prof.step()

    for root, _, files in os.walk(torch_path):
        for file in files:
            if file != 'kernel_details.csv':
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            filtered_df = filter_rows(df, filter_list)
            time_cost_df = filtered_df.groupby('Name')['Duration(us)'].mean().to_dict()
            time_cost_df['__total_us__'] = float(filtered_df['Duration(us)'].sum() / max(active, 1))
            delete_file(torch_path)
            return time_cost_df

    delete_file(torch_path)
    return {}
