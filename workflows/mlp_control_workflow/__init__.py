from pathlib import Path
import sys
import os
import time
from datetime import datetime
from workflows.shared.workflow_executor import create_workflow_logger

# Add project root to path for shared utilities
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Create logs/ directory if it doesn't exist
os.makedirs("./logs", exist_ok=True)

# Create a timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = time.time()
log_file_path = f"./logs/mlp_workflow_log_{timestamp}.txt"

# Redirect stdout and stderr to both terminal and log file
sys.stdout = create_workflow_logger(log_file_path)
sys.stderr = sys.stdout
