import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='eu-west-1')

bucket = 'dreem-dodo-dodh'
prefix = 'dod-h/'


script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'DOD-H')
os.makedirs(output_dir, exist_ok=True)


response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
files = response.get('Contents', [])


for i, file in enumerate(files, 1):
    key = file['Key']
    filename = key.split('/')[-1]
    if not filename:
        continue
    dest_path = os.path.join(output_dir, filename)
    print(f'Downloading {filename} ({i}/{len(files)})')
    s3.download_file(bucket, key, dest_path)
