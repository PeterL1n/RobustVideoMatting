from app.aws_processor import AWSProcessor
import os
import platform
from app.utilities import now, creation_date, logger
from os.path import exists
import string
import time
import random
from app.bg import removal


if __name__ == '__main__':
    if platform.system() != 'Windows':
        os.chdir('/home/ubuntu/lipsync')

    if not exists('app/logs/{}.log'.format(now(True))):
        with open('app/logs/{}.log'.format(now(True)), 'w') as f:
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
            time.sleep(120)
            exit(0)

    process_name = ''.join(random.choice(letters) for i in range(10))
    process_name = "{}.txt".format(process_name)

    # read sqs
    try:
        sqs, handler = aws_client.get_sqs(process_name)

        if not sqs:
            # todo check if there is a need of a balancer

            time.sleep(10)
            exit(0)

        uid = sqs['uid']
        logger().info('Bg removal process started for {}'.format(uid))
        # todo check if there is a need of a balancer create process file

        # create unique folder
        os.makedirs('app/files/{}'.format(uid), exist_ok=True)
        logger(uid).info('created folder: files/{}'.format(uid))

        # download video
        video = aws_client.download_video(sqs['video'], uid)
        logger(uid).info('downloaded video into: files/{}/video.mp4'.format(uid))

        # bg_removal

    except Exception as E:
        pass

    # bg_removal
    # upload video
    # send sqs
    # clear temporary data