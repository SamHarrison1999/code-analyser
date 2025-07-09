import logging
import math
import subprocess
import re
from typing import Any


def run_lizard(file_path: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["lizard", file_path],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        output = result.stdout
    except Exception as e:
        logging.error(f"Lizard error for {file_path}: {e}")
        return [
            {"name": "lizard_error", "value": None, "units": None, "success": False, "error": str(e)}
        ]

    cc_list, token_counts, parameter_counts, function_lengths, function_nloc_list = [], [], [], [], []
    file_nloc = 0
    func_regex = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+.+$")
    summary_regex = re.compile(r"Total:\s+\d+\s+functions,\s+(\d+)\s+NLOC,.*", re.IGNORECASE)

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        func_match = func_regex.match(line)
        if func_match:
            try:
                func_nloc = int(func_match.group(1))
                cc = int(func_match.group(2))
                tokens = int(func_match.group(3))
                params = int(func_match.group(4))
                length = int(func_match.group(5))
            except ValueError:
                continue
            function_nloc_list.append(func_nloc)
            cc_list.append(cc)
            token_counts.append(tokens)
            parameter_counts.append(params)
            function_lengths.append(length)
            continue

        summary_match = summary_regex.search(line)
        if summary_match:
            try:
                file_nloc = int(summary_match.group(1))
            except ValueError:
                file_nloc = 0

    if file_nloc == 0:
        file_nloc = sum(function_nloc_list)

    total_functions = len(cc_list)
    average_cc = sum(cc_list) / total_functions if total_functions > 0 else 0.0
    average_token_count = sum(token_counts) / total_functions if total_functions > 0 else 0.0
    average_parameter_count = sum(parameter_counts) / total_functions if total_functions > 0 else 0.0
    average_func_length = sum(function_lengths) / total_functions if total_functions > 0 else 0.0
    total_tokens = sum(token_counts)

    try:
        if file_nloc > 0 and total_tokens > 0:
            mi = (171 - 5.2 * math.log(total_tokens) - 0.23 * average_cc - 16.2 * math.log(file_nloc)) * 100 / 171
            mi = max(0, mi)
        else:
            mi = 0.0
    except Exception as e:
        logging.error(f"Error calculating MI for {file_path}: {e}")
        mi = 0.0

    logging.info(
        f'File: {file_path}\n'
        f'Average function complexity: {average_cc:.2f}\n'
        f'Average Token Count: {average_token_count:.2f}\n'
        f'Average Parameter Count: {average_parameter_count:.2f}\n'
        f'Average Function Length: {average_func_length:.2f}\n'
        f'Number of Functions: {total_functions}\n'
        f'Maintainability Index: {mi:.2f}'
    )

    return [
        {"name": "average_function_complexity", "value": average_cc, "units": None, "success": True, "error": None},
        {"name": "average_token_count", "value": average_token_count, "units": None, "success": True, "error": None},
        {"name": "average_parameter_count", "value": average_parameter_count, "units": None, "success": True, "error": None},
        {"name": "average_function_length", "value": average_func_length, "units": "lines", "success": True, "error": None},
        {"name": "number_of_functions", "value": total_functions, "units": "functions", "success": True, "error": None},
        {"name": "maintainability_index", "value": mi, "units": "score", "success": True, "error": None}
    ]


def gather_lizard_metrics(file_path: str) -> list[dict[str, Any]]:
    return run_lizard(file_path)
