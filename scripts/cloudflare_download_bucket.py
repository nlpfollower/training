import os
import fire
from config import get_config, update_config
from src.utils.cloudflare import CloudflareR2
from src.utils.logger import log


def download_bucket(bucket_name: str, download_path: str, config_updates: dict = None):
    # Get and update configuration
    config = get_config()
    if config_updates:
        update_config(config, **config_updates)

    # Initialize CloudflareR2 client
    cloudflare_r2 = CloudflareR2(config)

    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # List all objects in the bucket
    try:
        objects = cloudflare_r2.s3_client.list_objects_v2(Bucket=bucket_name)
    except Exception as e:
        log.error(f"Error listing objects in bucket {bucket_name}: {e}")
        return

    if 'Contents' not in objects:
        log.info(f"Bucket {bucket_name} is empty")
        return

    # Prepare files to download
    files_to_download = []
    for obj in objects['Contents']:
        object_name = obj['Key']
        file_path = os.path.join(download_path, object_name)

        # Ensure the directory exists for the file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        files_to_download.append((file_path, object_name))

    # Download files
    log.info(f"Downloading files from {bucket_name}")
    cloudflare_r2.download_files_concurrently(files_to_download, bucket_name)
    log.info(f"Downloaded all files from {bucket_name} to {download_path}")


def main(bucket_name: str, download_path: str, **kwargs):
    log.info(f"Starting download from bucket '{bucket_name}' to '{download_path}'")
    download_bucket(bucket_name, download_path, config_updates=kwargs)


if __name__ == "__main__":
    fire.Fire(main)