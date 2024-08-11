import threading
import time
from src.utils.logger import log_upload_progress

class UploadTracker:
    def __init__(self, total_files, total_size, bandwidth_monitor):
        self.total_files = total_files
        self.total_size = total_size
        self.bandwidth_monitor = bandwidth_monitor
        self.lock = threading.Lock()
        self.files_completed = 0
        self.bytes_transferred = 0
        self.last_log_time = time.time()
        self.log_interval = 5  # Log every 5 seconds

    def update(self, filename, bytes_amount):
        with self.lock:
            self.bytes_transferred += bytes_amount
            self.bandwidth_monitor.add_bytes(bytes_amount)
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                self.log_progress()
                self.last_log_time = current_time

    def complete_file(self):
        with self.lock:
            self.files_completed += 1
            self.log_progress()

    def log_progress(self):
        log_upload_progress(self.files_completed, self.total_files, self.bytes_transferred, self.total_size)
