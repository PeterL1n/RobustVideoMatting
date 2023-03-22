import json
import boto3
from typing import Union, Tuple, Any
from app.settings import AWS_CONFIG
import os
from app.utilities import now, generate_video_path, generate_final_video, generate_s3_video_arn, generate_s3_media_arn
from botocore.exceptions import ClientError
import requests
import time
# from balancer import Balancer
# import shutil

class AWSProcessor:
    def __init__(self):
        self.sqs_client = boto3.client(
            'sqs', aws_access_key_id=AWS_CONFIG['key'], aws_secret_access_key=AWS_CONFIG['secret'], region_name=AWS_CONFIG['region'])
        self.s3_client = boto3.client(
            's3', aws_access_key_id=AWS_CONFIG['key'], aws_secret_access_key=AWS_CONFIG['secret'])

        self.bucket = AWS_CONFIG['bucket']
        self.sqs_url = AWS_CONFIG['sqs']

    def get_sqs_client(self):
        return self.sqs_client

    def get_s3_client(self):
        return self.s3_client

    def upload_logs(self, uid=None):
        log_general = "app/logs/{}.log".format(now(True))
        job_log = False
        if uid is not None:
            job_log = "app/logs/{}.log".format(uid)
        try:
            self.s3_client.upload_file(log_general, self.bucket, "logs/bg-removal/{}".format(os.path.basename(log_general)))
            if uid is not None and job_log:
                self.s3_client.upload_file(job_log, self.bucket, "logs/bg-removal/{}".format(os.path.basename(job_log)))
        except ClientError as E:
            raise Exception(E)

    def uplaod_final_video(self, uid, final_video_local):
        try:
            _, extension = os.path.splitext(final_video_local)
            self.s3_client.upload_file(final_video_local, self.bucket, "tmp/{}{}".format(uid, extension), ExtraArgs={'ACL': 'public-read'})
            return "https://{}.s3.amazonaws.com/tmp-lip-sync-avatar/{}{}".format(self.bucket, uid, extension)
        except ClientError as E:
            raise Exception(E)

    def delete_sqs_message(self, handler):
        self.sqs_client.delete_message(
            QueueUrl=self.sqs_url,
            ReceiptHandle=handler
        )


    def get_sqs(self, process_name) -> Union[Tuple[Any, Any], bool]:
        """
        This method is responsible for reading AWS SQS queues through aws_process   or
        Returns:
            Union[dict, bool]:
        """
        # balancer = Balancer('lipsync')
        # if balancer.is_main():
        #     balancer.create_blocker(process_name)
        # elif balancer.main_running():
        #     time.sleep(30)
        #     return False, False
        # elif not balancer.can_run():
        #     time.sleep(30)
        #     return False, False

        response = self.sqs_client.receive_message(QueueUrl=self.sqs_url, MaxNumberOfMessages=1, WaitTimeSeconds=2)

        for message in response.get('Messages', []):
            message_body = message['Body']
            sqs_message_handler = message['ReceiptHandle']
            # while not balancer.can_run():
            #     time.sleep(10)
            #     continue

            return json.loads(message_body), sqs_message_handler

        # balancer.remove_process(process_name)
        return '', ''

    def generate_full_url(self, arn):
        return "https://{}.s3.amazonaws.com/tts/{}.wav".format(self.bucket, arn)

    def download_video(self, video, uid):
        bucket, arn = generate_s3_media_arn(video)
        _, extension = os.path.splitext(video)
        video_local = generate_video_path(uid, extension)

        try:
            self.s3_client.download_file(bucket, arn, video_local)
        except Exception as E:
            print(E)
        return video_local, extension

    def file_exists(self, arn):
        try:
            self.s3_client.head_object(Bucket='mltts', Key=arn)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            else:
                return False
        else:
            return True