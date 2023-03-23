from app.aws_processor import AWSProcessor
import os
import platform
from app.utilities import now, creation_date, logger, send_webhook, shutdown_single_proccess
import logging
from os.path import exists
import string
import time
import random
from app.bg import removal
import shutil
import traceback


if __name__ == '__main__':
    if platform.system() != 'Windows':
        os.chdir('/home/ubuntu/videomatting')

    instance_id = os.popen('wget -q -O - http://169.254.169.254/latest/meta-data/instance-id').read()
    if not exists('app/logs/{}_{}.log'.format(now(True), instance_id)):
        with open('app/logs/{}_{}.log'.format(now(True), instance_id), 'w') as f:
            f.write(now())

    # general config
    letters = string.ascii_lowercase
    aws_client = AWSProcessor()

    # check server status for asgs
    if exists('/home/ubuntu/terminate.txt'):
        time_now = time.time()
        created_termination = creation_date('/home/ubuntu/terminate.txt')
        minutes_pass = (time_now - created_termination) / 60
        if minutes_pass > 15:
            os.remove('/home/ubuntu/terminate.txt')
        else:
            time.sleep(20)
            exit(0)

    process_name = ''.join(random.choice(letters) for i in range(10))
    process_name = "{}.txt".format(process_name)

    # read sqs
    try:
        sqs, handler = aws_client.get_sqs(process_name)
        #
        # sqs = {
        #     'uid': 'sdasda',
        #     'video': 'https://mltts.s3.amazonaws.com/tmp/test.mp4'
        # }

        if not sqs:
            shutdown_single_proccess()
            # todo check if there is
            #  a need of a balancer
            time.sleep(10)
            exit(0)

        uid = sqs['uid']
        logger().info('Bg removal process started for {}'.format(uid))
        # todo check if there is a need of a balancer create process file

        logger(uid).info('{} task running on {} server'.format(uid, instance_id))

        # create unique folder
        os.makedirs('app/files/{}'.format(uid), exist_ok=True)
        logger(uid).info('created folder: files/{}'.format(uid))

        # download video
        video, extension = aws_client.download_video(sqs['video'], uid)
        logger(uid).info('downloaded video into: files/{}/video.mp4'.format(uid))

        # bg_removal
        local_file = removal(uid, extension=extension)

        # upload final video
        logger(uid).info('uploading final video ')
        final_video_url = aws_client.uplaod_final_video(uid, local_file)
        print(final_video_url)

        # webhook
        send_webhook(data={'video': final_video_url, 'uid': uid})

        # delete sqs
        aws_client.delete_sqs_message(handler)

        # upload logs
        aws_client.upload_logs(uid=uid)

        # clear data
        handlers = logging.getLogger(uid).handlers[:]
        for handler in handlers:
            handler.close()
        os.remove("app/logs/{}.log".format(uid))

        # remove files
        time.sleep(10)
        shutil.rmtree('app/files/{}'.format(uid))
        shutil.rmtree('temp/{}'.format(uid))



    except Exception as E:
        print(E)
        print(traceback.format_exc())
        pass