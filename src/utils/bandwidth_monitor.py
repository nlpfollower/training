import threading
import time
from src.utils.logger import log_bandwidth


class BandwidthMonitor:
    def __init__(self, interval=10):  # Changed to 10 seconds
        self.interval = interval
        self.lock = threading.Lock()
        self.total_bytes = 0
        self.last_bytes = 0
        self.start_time = time.time()

    def add_bytes(self, byte_count):
        with self.lock:
            self.total_bytes += byte_count

    def monitor(self):
        while True:
            time.sleep(self.interval)
            with self.lock:
                current_bytes = self.total_bytes
                bytes_since_last = current_bytes - self.last_bytes
                self.last_bytes = current_bytes

            current_time = time.time()
            elapsed_time = current_time - self.start_time
            avg_speed = self.total_bytes / elapsed_time / 1024 / 1024  # MB/s
            current_speed = bytes_since_last / self.interval / 1024 / 1024  # MB/s
            log_bandwidth(current_speed, avg_speed)
