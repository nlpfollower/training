import os
import fire
import pydevd_pycharm

from config import get_config, update_config
from src.utils.cloudflare import CloudflareR2
from src.utils.logger import log


def prepare_boot(name: str, num_buckets: int, config_updates: dict = None):
    # Get and update configuration
    config = get_config()
    if config_updates:
        update_config(config, **config_updates)

    # Initialize CloudflareR2 client
    cloudflare_r2 = CloudflareR2(config)

    # Path to the boot.tar file
    boot_file_path = "/root/boot.tar"

    # Check if boot.tar exists
    if not os.path.exists(boot_file_path):
        log.bind(custom_label="error").error(f"boot.tar not found at {boot_file_path}")
        return

    # Create buckets and upload boot.tar
    for i in range(num_buckets):
        bucket_name = f"{name}-{i}"

        # Create bucket
        try:
            cloudflare_r2.s3_client.head_bucket(Bucket=bucket_name)
            log.bind(custom_label="info").info(f"Bucket {bucket_name} already exists")
        except:
            cloudflare_r2.s3_client.create_bucket(Bucket=bucket_name)
            log.bind(custom_label="info").info(f"Created bucket {bucket_name}")

        # Upload boot.tar
        log.bind(custom_label="info").info(f"Uploading boot.tar to {bucket_name}")
        files_to_upload = [(boot_file_path, "boot.tar")]
        cloudflare_r2.upload_files_concurrently(files_to_upload, bucket_name)
        log.bind(custom_label="success").info(f"Uploaded boot.tar to {bucket_name}")

    log.bind(custom_label="success").info(f"Completed preparation for {num_buckets} buckets")


def main(name: str, num_buckets: int, **kwargs):
    log.bind(custom_label="main").info(f"Starting boot preparation for {num_buckets} buckets with name prefix '{name}'")
    prepare_boot(name, num_buckets, config_updates=kwargs)


if __name__ == "__main__":
    #pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    fire.Fire(main)