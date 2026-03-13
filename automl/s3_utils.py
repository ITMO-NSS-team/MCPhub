import os
from io import BytesIO
from pathlib import Path
from typing import List

import boto3
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()


class S3BucketService:
    """Service for basic operations with an S3-compatible bucket."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str = "default",
    ) -> None:
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key

    def create_s3_client(self):
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
        )

    def upload_file_object(
        self,
        prefix: str,
        source_file_name: str,
        file_path: str,
    ) -> None:
        client = self.create_s3_client()
        destination_path = Path(prefix, source_file_name).as_posix()

        with open(file_path, "rb") as file_obj:
            content = file_obj.read()

        buffer = BytesIO(content)
        client.upload_fileobj(buffer, self.bucket_name, destination_path)

    def list_objects(self, prefix: str = "") -> List[str]:
        client = self.create_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

        storage_content: List[str] = []
        for page in page_iterator:
            contents = page.get("Contents", [])
            for item in contents:
                storage_content.append(item["Key"])

        return storage_content

    def delete_file_object(self, prefix: str, source_file_name: str) -> None:
        client = self.create_s3_client()
        path_to_file = Path(prefix, source_file_name).as_posix()
        client.delete_object(Bucket=self.bucket_name, Key=path_to_file)

    def create_new_bucket(self, bucket_name: str) -> None:
        client = self.create_s3_client()
        client.create_bucket(Bucket=bucket_name)

    def del_bucket(self, bucket_name: str) -> None:
        client = self.create_s3_client()
        client.delete_bucket(Bucket=bucket_name)

    def generate_presigned_url(
        self,
        s3_key: str,
        method: str = "get_object",
        expiration: int = 360,
    ) -> str:
        client = self.create_s3_client()
        return client.generate_presigned_url(
            method,
            Params={"Bucket": self.bucket_name, "Key": s3_key},
            ExpiresIn=expiration,
        )

    def download_image_from_s3(self, s3_key: str, local_path: str) -> None:
        client = self.create_s3_client()
        client.download_file(self.bucket_name, s3_key, local_path)

    def get_image_bytes_from_s3(self, s3_key: str, bucket_name: str) -> bytes:
        client = self.create_s3_client()
        response = client.get_object(Bucket=bucket_name, Key=s3_key)
        return response["Body"].read()

    def clean_up_by_prefix(self, prefix_to_delete: str) -> None:
        client = self.create_s3_client()
        response = client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix_to_delete)
        contents = response.get("Contents", [])
        for item in contents:
            client.delete_object(Bucket=self.bucket_name, Key=item["Key"])


s3_service = S3BucketService(
    endpoint=(os.getenv("ENDPOINT_URL") or "").strip(),
    access_key=(os.getenv("ACCESS_KEY") or "").strip(),
    secret_key=(os.getenv("SECRET_KEY") or "").strip(),
    bucket_name=(os.getenv("BUCKET_NAME") or "default").strip(),
)
