import logging
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
from app.settings import WEBHOOK_CONFIG, SHUTDOWN_TIMINGS, SERVER
import glob
import time
import platform
load_dotenv()


def send_webhook(url=None, headers=None, data=None) -> bool:

    if url is None:
        url = WEBHOOK_CONFIG['url']

    if headers is None:
        headers = {
            'Authorization': WEBHOOK_CONFIG['key']
        }

    try:
        if data is None:
            requests.get(url, headers=headers)
        else:
            requests.post(url, data=data, headers=headers, timeout=10)
    except Exception as E:
        logger().error(E)


def logger(file=None):
    file_name = file
    if file is None:
        instance_id = os.popen('wget -q -O - http://169.254.169.254/latest/meta-data/instance-id').read()
        daily_log = "{}_{}.log".format(now(True), instance_id)
        file = "app/logs/{}".format(daily_log)
        level = 'DEBUG'
    else:
        file = "app/logs/{}.log".format(file)
        level = 'INFO'

    log_format = logging.Formatter("%(levelname)s %(asctime)s - %(message)s")

    handler = logging.FileHandler(file)
    handler.setFormatter(log_format)
    logger = logging.getLogger(file_name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    return logger


def clear_old_data(file_or_dir, is_folder=False):
    if is_folder:
        files = glob.glob(file_or_dir)
        for f in files:
            os.remove(f)
    else:
        if os.path.exists(file_or_dir):
            os.remove(file_or_dir)


def now(is_only_date=False):
    return datetime.now().strftime("%Y-%m-%d") if is_only_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_s3_media_arn(media):
    bucket_full, arn = media.split('/', 2)[-1].split('/', 1)
    return bucket_full.split('.')[0], arn


def generate_s3_video_arn(avatar_id):
    return 'avatars/{}.mp4'.format(avatar_id)


def generate_video_path(uid, extension):
    return "app/files/{}/video{}".format(uid, extension)


def generate_final_video(uid, extension):
    return 'app/files/{}/final{}'.format(uid, extension)


def shutdown_single_proccess():
    instance_id = os.popen('wget -q -O - http://169.254.169.254/latest/meta-data/instance-id').read()

    if instance_id == SERVER['main_id']:
        return False

    if not check_runtime():
        return False

    if not other_servers_are_running():
        return False

    with open('/home/ubuntu/terminate.txt', 'w') as f:
        f.write(now())

    terminate_command = "aws autoscaling terminate-instance-in-auto-scaling-group --instance-id {} --should-decrement-desired-capacity 2>&1".format(
        instance_id)
    os.system(terminate_command)


def shutdown():
    instance_id = os.popen('wget -q -O - http://169.254.169.254/latest/meta-data/instance-id').read()
    if instance_id == SERVER['main_id']:
        return False
    else:
        if not check_runtime():
            return False

        if not check_all_processes_statuses():
            return False

        with open('/home/ubuntu/terminate.txt', 'w') as f:
            f.write(now())

        terminate_command = "aws autoscaling terminate-instance-in-auto-scaling-group --instance-id {} --should-decrement-desired-capacity 2>&1".format(
            instance_id)
        os.system(terminate_command)


def other_servers_are_running():
    import json

    command = 'aws autoscaling describe-auto-scaling-groups --auto-scaling-group-name ASG-ML-BACKGROUND-REMOVAL'
    output = os.popen(command).read()
    asg_info = json.loads(output)

    min_size = asg_info['AutoScalingGroups'][0]['MinSize']
    if min_size == 0:
        return True

    instances_info = asg_info['AutoScalingGroups'][0]['Instances']
    if len(instances_info) <= min_size:
        return False

    return True


def check_runtime():
    time_info = os.popen('who -b').read()
    time = int(time_info.split(':')[-1])
    terminate_time = time + int(SHUTDOWN_TIMINGS['minutes']) - int(SHUTDOWN_TIMINGS['intermediate'])
    terminate_time = terminate_time if terminate_time < 60 else terminate_time - 60

    time_now = int(datetime.now().strftime("%M"))
    if terminate_time - 1 <= time_now <= terminate_time + 1:
        return True

    return False


def check_all_processes_statuses():
    from settings import BALANCER_SERVER_TYPES
    time_now = time.time()
    for proc in BALANCER_SERVER_TYPES:
        if not len(os.listdir('/home/ubuntu/processes/{}'.format(proc))) == 0:
            processes = os.listdir('/home/ubuntu/processes/{}'.format(proc))
            for process in processes:
                created = creation_date('/home/ubuntu/processes/{}/{}'.format(proc, process))
                minutes_pass = (time_now - created) / 60
                if minutes_pass > 50:
                    os.remove('process/{}'.format(process))
                else:
                    return False

    return True


def creation_date(file):
    if platform.system() == 'Windows':
        return os.path.getctime(file)
    else:
        stat = os.stat(file)
        try:
            return stat.st_ctime
        except AttributeError:
            return stat.st_mtime

def check_processess():
    import time
    time_now = time.time()
    processes = os.listdir('process')
    for process in processes:
        created = creation_date('process/{}'.format(process))
        minutes_pass = (time_now - created)/60
        if minutes_pass > 15:
            os.remove('process/{}'.format(process))
        else:
            return False

    return True