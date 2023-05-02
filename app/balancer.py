import os

from settings import BALANCER, BALANCER_TIMES, BALANCER_SERVER_TYPES, BALANCER_RESOURCES, SERVER_RESOURCES_TOTAL
from utilities import now
from os.path import exists
from datetime import datetime
import nvidia_smi
import psutil
import time


class Balancer:
    def __init__(self, current_server):
        self.current_server = current_server

    def is_main(self):
        return self.current_server == BALANCER['main_for']

    def create_blocker(self, process_name):
        with open('/home/ubuntu/processes/{}/{}'.format(self.current_server, process_name), 'w+') as f:
            f.write(now())

    def main_running(self):
        main_server_processes = len(os.listdir('/home/ubuntu/processes/{}'.format(BALANCER['main_for'])))
        return main_server_processes != 0

    def can_run_main(self):
        processes = 0
        for proc in BALANCER_SERVER_TYPES:
            if proc == self.current_server:
                continue
            processes = len(os.listdir('/home/ubuntu/processes/{}'.format(proc))) + \
                        len(os.listdir('/home/ubuntu/processes/{}'.format(proc))) + \
                        len(os.listdir('/home/ubuntu/processes/{}'.format(proc)))

        print("processes ", processes)
        if processes == 0:
            return True

        return False

    def can_run(self):
        if self.is_main():
            print('main check')
            return self.can_run_main()
        print('has time {}'.format(self.has_time()))
        print('has resource {}'.format(self.has_resource()))
        return self.has_time() and self.has_resource()

    def remove_process(self, process_name):
        process_filename = '/home/ubuntu/processes/{}/{}'.format(self.current_server, process_name)
        if exists(process_filename):
            os.remove(process_filename)

    def has_time(self):
        time_info = os.popen('who -b').read()
        time = int(time_info.split(':')[-1]) # 22 sarqvela
        time_now = int(datetime.now().strftime("%M"))
        remaining_time = time + 55 - time_now
        return remaining_time >= BALANCER_TIMES[self.current_server]

    def has_resource(self):
        cpu_use = 0
        ram_use = 0
        gpu_use = 0
        nvidia_smi.nvmlInit()

        for i in range(60):
            cpu_use += psutil.cpu_percent()

            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            gpu_use += info.free / (1024 ** 2)
            ram_use += psutil.virtual_memory().free / (1024 ** 2)

            time.sleep(0.5)

        gpu = int(gpu_use / 60)
        ram = int(ram_use / 60)
        cpu = int(cpu_use / 60)

        if gpu < BALANCER_RESOURCES[self.current_server]['gpu'] or ram < BALANCER_RESOURCES[self.current_server]['ram']:
            return False

        gpu, ram, cpu = self.other_processes_status()
        if gpu < BALANCER_RESOURCES[self.current_server]['gpu'] or ram < BALANCER_RESOURCES[self.current_server]['ram']:
            return False

        return True

    def other_processes_status(self):
        gpu_free_proc = int(SERVER_RESOURCES_TOTAL['gpu'])
        ram_free_proc = int(SERVER_RESOURCES_TOTAL['ram'])
        cpu_free_proc = 100

        for proc in BALANCER_SERVER_TYPES:
            if proc == self.current_server:
                continue
            gpu_free_proc -= int(len(os.listdir('/home/ubuntu/processes/{}'.format(proc)))) * 1.2 * int(BALANCER_RESOURCES[proc]['gpu'])
            ram_free_proc -= int(len(os.listdir('/home/ubuntu/processes/{}'.format(proc)))) * 1.2 * int(BALANCER_RESOURCES[proc]['ram'])
            # cpu_free_proc -= int(len(os.listdir('/home/ubuntu/processes/{}'.format(proc)))) * 1.2 * BALANCER_RESOURCES[proc]['cpu']

        return gpu_free_proc, ram_free_proc, cpu_free_proc
