import os
import fire
from config import get_config, update_config
from src.utils.cloudflare import CloudflareR2
from src.utils.logger import log

def prepare_boot(bucket_name: str, config_updates: dict = None):
    # Get and update configuration
    config = get_config()
    if config_updates:
        update_config(config, **config_updates)

    # Initialize CloudflareR2 client
    cloudflare_r2 = CloudflareR2(config)

    # Path to the boot.tar file
    boot_file_path = "/root/boot.tar"

    # Path to the model directory
    model_path = "/root/model"

    # Check if boot.tar exists
    if not os.path.exists(boot_file_path):
        log.error(f"boot.tar not found at {boot_file_path}")
        return

    # Check if model directory exists
    if not os.path.exists(model_path):
        log.error(f"Model directory not found at {model_path}")
        return

    # Create bucket if it doesn't exist
    try:
        cloudflare_r2.s3_client.head_bucket(Bucket=bucket_name)
        log.info(f"Bucket {bucket_name} already exists")
    except:
        cloudflare_r2.s3_client.create_bucket(Bucket=bucket_name)
        log.info(f"Created bucket {bucket_name}")

    # Prepare files to upload
    files_to_upload = []

    # Add boot.tar
    files_to_upload.append((boot_file_path, "boot.tar", bucket_name))

    # Add all files from model_path
    for root, _, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, model_path)
            files_to_upload.append((file_path, relative_path, bucket_name))

    # Upload files
    log.info(f"Uploading files to {bucket_name}")
    cloudflare_r2.upload_files_concurrently(files_to_upload)
    log.info(f"Uploaded all files to {bucket_name}")

    log.info(f"Completed preparation for bucket {bucket_name}")

def main(bucket_name: str, **kwargs):
    log.info(f"Starting boot preparation for bucket '{bucket_name}'")
    prepare_boot(bucket_name, config_updates=kwargs)
    log.info(f"Completed boot preparation for bucket '{bucket_name}'")

if __name__ == "__main__":
    fire.Fire(main)