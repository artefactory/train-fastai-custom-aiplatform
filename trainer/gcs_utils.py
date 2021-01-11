import subprocess
from google.cloud import storage

def download_file_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

def upload_file_to_gcs(local_file_name, gcs_file_path):
    subprocess.check_call([
            'gsutil', 'cp', local_file_name, gcs_file_path])
