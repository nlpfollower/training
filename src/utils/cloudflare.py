import os
import threading
import time
import concurrent.futures
from typing import List, Tuple

from botocore.exceptions import ClientError
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config

from src.utils.bandwidth_monitor import BandwidthMonitor
from src.utils.logger import log
from src.utils.upload_progress import UploadTracker


class CloudflareR2:
    def __init__(self, config):
        self.config = config
        self.s3_client = self._create_s3_client()

    def _create_s3_client(self):
        config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        return boto3.client('s3',
                            endpoint_url=self.config.api.cloudflare_endpoint_url,
                            aws_access_key_id=self.config.api.cloudflare_access_key_id,
                            aws_secret_access_key=self.config.api.cloudflare_secret_access_key,
                            config=config)

    def upload_file(self, file_name, object_name, bucket_name, progress_tracker):
        if object_name is None:
            object_name = os.path.basename(file_name)

        transfer_config = TransferConfig(
            multipart_threshold=self.config.cloudflare.multipart_threshold,
            max_concurrency=self.config.cloudflare.max_concurrency,
            multipart_chunksize=self.config.cloudflare.multipart_chunksize,
            use_threads=True
        )

        try:
            self.s3_client.upload_file(
                file_name, bucket_name, object_name,
                Config=transfer_config,
                Callback=lambda bytes_amount: progress_tracker.update(file_name, bytes_amount),
                ExtraArgs={'StorageClass': 'STANDARD'}
            )
            progress_tracker.complete_file()
        except ClientError as e:
            log.bind(custom_label="error").error(f"Error uploading {file_name}: {e}")
            return False
        return True

    def upload_files_concurrently(self, files: List[Tuple[str, str, str]]):
        total_size = sum(os.path.getsize(data_file) for data_file, _, _ in files)
        bandwidth_monitor = BandwidthMonitor()
        progress_tracker = UploadTracker(len(files), total_size, bandwidth_monitor)

        monitor_thread = threading.Thread(target=bandwidth_monitor.monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.cloudflare.max_concurrency) as executor:
            future_to_file = {executor.submit(self.upload_file, data_file, chunk_file, bucket_name, progress_tracker):
                (data_file, chunk_file, bucket_name) for data_file, chunk_file, bucket_name in files if os.path.exists(data_file)}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    success = future.result()
                    if not success:
                        log.bind(custom_label="warning").warning(f"Failed to upload {file}")
                except Exception as e:
                    log.bind(custom_label="error").error(f"Exception occurred while uploading {file}: {e}")

    def download_file(self, bucket_name, object_name, file_name, progress_tracker):
        transfer_config = TransferConfig(
            multipart_threshold=self.config.cloudflare.multipart_threshold,
            max_concurrency=self.config.cloudflare.max_concurrency,
            multipart_chunksize=self.config.cloudflare.multipart_chunksize,
            use_threads=True
        )

        try:
            self.s3_client.download_file(
                bucket_name, object_name, file_name,
                Config=transfer_config,
                Callback=lambda bytes_amount: progress_tracker.update(file_name, bytes_amount)
            )
            progress_tracker.complete_file()
        except ClientError as e:
            log.bind(custom_label="error").error(f"Error downloading {object_name}: {e}")
            return False
        return True

    def download_files_concurrently(self, files, bucket_name):
        total_size = sum(self.s3_client.head_object(Bucket=bucket_name, Key=object_name)['ContentLength']
                         for _, object_name in files)
        bandwidth_monitor = BandwidthMonitor()
        progress_tracker = UploadTracker(len(files), total_size, bandwidth_monitor)

        monitor_thread = threading.Thread(target=bandwidth_monitor.monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.cloudflare.max_concurrency) as executor:
            future_to_file = {executor.submit(self.download_file, bucket_name, object_name, file_name, progress_tracker): (file_name, object_name)
                for file_name, object_name in files}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    success = future.result()
                    if not success:
                        log.bind(custom_label="warning").warning(f"Failed to download {file}")
                except Exception as e:
                    log.bind(custom_label="error").error(f"Exception occurred while downloading {file}: {e}")