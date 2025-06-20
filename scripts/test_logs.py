# Script to check if logs/ has activity
import os

log_dir = "logs"
if not os.path.exists(log_dir):
    print("No logs found.")
else:
    logs = os.listdir(log_dir)
    print(f"{len(logs)} log(s) found.")