#!/usr/bin/env python3
"""
Shared Workflow Executor

Common workflow execution logic for both MLP and CTC workflows.
Provides step-by-step execution with timing, error handling, and logging.
"""

import time
import traceback
from typing import List, Tuple, Callable


def execute_workflow_steps(steps: List[Tuple[str, Callable, ...]], workflow_name: str = "Workflow") -> float:
    """
    Execute a list of workflow steps with timing and error handling.

    Args:
        steps: List of tuples containing (label, function, *optional_params)
        workflow_name: Name of the workflow for logging purposes
    """
    print(f"🚀 Starting {workflow_name} execution...")
    print("=" * 60)

    total_start_time = time.time()

    for step_data in steps:
        if len(step_data) < 2:
            print(f"⚠️ Invalid step format: {step_data}")
            continue

        label = step_data[0]
        func = step_data[1]
        params = step_data[2:] if len(step_data) > 2 else []

        print(f"\n🚀 Starting: {label}...")
        step_start = time.time()

        try:
            if params:
                func(*params)
            else:
                func()
        except Exception as e:
            print(f"❌ Failed: {label} - {e}")
            traceback.print_exc()
            # Continue with next step for robustness

        step_end = time.time()
        duration = step_end - step_start
        print(f"✅ Finished: {label} in {duration:.2f} seconds\n")

    total_time = time.time() - total_start_time
    print(f"\n\n🏁 {workflow_name} completed! 🏁")
    print("=" * 40)
    print(f"✅✅ {workflow_name} complete! Total time: {total_time:.2f} seconds ✅✅")


def create_workflow_logger(log_file_path: str):
    """
    Create a logger that writes to both console and file.

    Args:
        log_file_path: Path to the log file

    Returns:
        Logger instance
    """
    import sys

    class Logger:
        def __init__(self, log_path):
            self.terminal = sys.__stdout__
            self.log = open(log_path, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)  # type: ignore
            self.log.write(message)

        def flush(self):
            self.terminal.flush()  # type: ignore
            self.log.flush()

    return Logger(log_file_path)
