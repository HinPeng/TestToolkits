import argparse
import json
import re

from pathlib import Path
from prettytable import PrettyTable


def parse_log_file(log_file_path):
    kernel_data = {}

    pattern_launcher_count = re.compile(r"(\w+)\s+candidate launcher count = (\d+)")
    pattern_precompile_time = re.compile(r"(\w+)\s+precompile elapsed time: ([\d.]+)")
    pattern_benchmark_time = re.compile(r"(\w+)\s+benchmark elapsed time: ([\d.]+)")

    with open(log_file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern_launcher_count.search(line)
            if match:
                kernel_name = match.group(1)
                count = int(match.group(2))
                kernel_data.setdefault(kernel_name, {})["launcher_count"] = count

            match = pattern_precompile_time.search(line)
            if match:
                kernel_name = match.group(1)
                kernel_data.setdefault(kernel_name, {})["precompile_time"] = float(
                    match.group(2)
                )

            match = pattern_benchmark_time.search(line)
            if match:
                kernel_name = match.group(1)
                assert kernel_name in kernel_data  # benchmark time must appear after precompile
                # only the first benchmark time is valid
                kernel_data[kernel_name].setdefault("benchmark_time", float(match.group(2)))

    log_file_path = Path(log_file_path)
    perf_file_path = log_file_path.parent / f"{log_file_path.stem}-perf.json"
    if perf_file_path.exists():
        with open(perf_file_path, "r", encoding="utf-8") as f:
            perf_dict = json.load(f)
        for kernel_name, perf in perf_dict.items():
            if kernel_name in kernel_data:
                kernel_data[kernel_name]["performance"] = perf
            else:
                # check if startswith key in kernel_data
                for k in kernel_data.keys():
                    if kernel_name.startswith(k) or k.startswith(kernel_name):
                        kernel_data[k]["performance"] = perf
                        break
    else:
        print(f"[Warning] Perf json file not exists: {perf_file_path}")

    return kernel_data


def print_table(kernel_data):
    table = PrettyTable()
    table.field_names = [
        "Kernel Name",
        "Config Number",
        "Autotune Time (s)",
        "Performance (us)",
    ]

    total_autotune_time = 0
    total_performance = 0
    kernel_count = 0

    for kernel_name, data in kernel_data.items():
        precompile_time = data.get('precompile_time', 0)
        benchmark_time = data.get('benchmark_time', 0)
        performance = data.get('performance', 0)
        autotune_time = precompile_time + benchmark_time

        total_autotune_time += autotune_time
        total_performance += performance
        kernel_count += 1

        table.add_row(
            [
                kernel_name,
                (
                    str(data["launcher_count"])
                    if data["launcher_count"] is not None
                    else "N/A"
                ),
                str(autotune_time) if autotune_time > 0 else "N/A",
                str(performance) if performance > 0 else "N/A",
            ]
        )

    if kernel_count > 1:
        table.add_row(
            ["Total", "N/A", str(total_autotune_time), str(total_performance)]
        )

    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str, help="Input log file path")
    args = parser.parse_args()

    kernel_data = parse_log_file(args.log_path)
    print_table(kernel_data)
