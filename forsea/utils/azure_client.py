import os
import uuid
import sys
import datetime
# from azure.storage.blob import BlockBlobService, PublicAccess, BlobServiceClient
from azure.storage.blob import PublicAccess, BlobServiceClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential

from typing import Optional, Iterable

def download_data(blob_folder_path: str, local_folder_path: str, container_name: str, connection_string: Optional[str]=None) -> int:
    try:        
        # Create the BlockBlobService that is used to call the Blob service for the storage account
        if connection_string is None:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container=container_name)
        
        if not os.path.exists(local_folder_path):
            os.makedirs(local_folder_path)

        generator = container_client.list_blobs(name_starts_with=blob_folder_path)
        files_downloaded = 0
        for blob in generator:
            blob_name: str = blob.name # type: ignore
            print("blob", blob.name)
            blob_path_split = blob_name.split('/')
            download_folder_path = os.path.join(local_folder_path, *blob_path_split[:-1])
            if not os.path.exists(download_folder_path):
                os.makedirs(download_folder_path)
            download_file_path = os.path.join(download_folder_path, blob_path_split[-1])
            print(download_folder_path)
            print(download_file_path)
            with open(file=download_file_path, mode="wb") as download_file:
                download_file.write(container_client.download_blob(blob_name).readall())
            files_downloaded += 1

        if files_downloaded > 0:
            sys.stdout.write("Finished downloading data.")
            sys.stdout.flush()
        else:
            sys.stdout.write("No data was downloaded.")
            sys.stdout.flush()
        return files_downloaded
    except Exception as e:
        print(e)
        return -1


def upload_data(blob_folder_path: str, file_list: str, container_name: str, local_folder_path: str, connection_string: Optional[str]=None) -> int:
    try:
        # Create the BlockBlobService that is used to call the Blob service for the storage account
        if connection_string is None:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        for file_name in file_list:
            print(f'Uploading to Azure Storage as blob: {os.path.join(local_folder_path, file_name)}')
            # Create a blob client using the local file name as the name for the blob
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_folder_path + file_name)

            upload_file_path = os.path.join(local_folder_path, file_name)
            # file = open(file=upload_file_path, mode='w')

            # Upload the created file
            with open(file=upload_file_path, mode="rb") as data:
                blob_client.upload_blob(data, overwrite=True)
        
        return True
                
    except Exception as e:
        print(e)
        return False