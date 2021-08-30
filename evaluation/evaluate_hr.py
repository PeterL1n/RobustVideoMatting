"""
HR (High-Resolution) evaluation. We found using numpy is very slow for high resolution, so we moved it to PyTorch using CUDA.

Note, the script only does evaluation. You will need to first inference yourself and save the results to disk
Expected directory format for both prediction and ground-truth is:

    videomatte_1920x1080
        ├── videomatte_motion
          ├── pha
            ├── 0000
              ├── 0000.png
          ├── fgr
            ├── 0000
              ├── 0000.png
        ├── videomatte_static
          ├── pha
            ├── 0000
              ├── 0000.png
          ├── fgr
            ├── 0000
              ├── 0000.png

Prediction must have the exact file structure and file name as the ground-truth,
meaning that if the ground-truth is png/jpg, prediction should be png/jpg.

Example usage:

python evaluate.py \
    --pred-dir pred/videomatte_1920x1080 \
    --true-dir true/videomatte_1920x1080
    
An excel sheet with evaluation results will be written to "pred/videomatte_1920x1080/videomatte_1920x1080.xlsx"
"""


import argparse
import os
import cv2
import kornia
import numpy as np
import xlsxwriter
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class Evaluator:
    def __init__(self):
        self.parse_args()
        self.init_metrics()
        self.evaluate()
        self.write_excel()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--pred-dir', type=str, required=True)
        parser.add_argument('--true-dir', type=str, required=True)
        parser.add_argument('--num-workers', type=int, default=48)
        parser.add_argument('--metrics', type=str, nargs='+', default=[
            'pha_mad', 'pha_mse', 'pha_grad', 'pha_dtssd', 'fgr_mse'])
        self.args = parser.parse_args()
        
    def init_metrics(self):
        self.mad = MetricMAD()
        self.mse = MetricMSE()
        self.grad = MetricGRAD()
        self.dtssd = MetricDTSSD()
        
    def evaluate(self):
        tasks = []
        position = 0
        
        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            for dataset in sorted(os.listdir(self.args.pred_dir)):
                if os.path.isdir(os.path.join(self.args.pred_dir, dataset)):
                    for clip in sorted(os.listdir(os.path.join(self.args.pred_dir, dataset))):
                        future = executor.submit(self.evaluate_worker, dataset, clip, position)
                        tasks.append((dataset, clip, future))
                        position += 1
                    
        self.results = [(dataset, clip, future.result()) for dataset, clip, future in tasks]
        
    def write_excel(self):
        workbook = xlsxwriter.Workbook(os.path.join(self.args.pred_dir, f'{os.path.basename(self.args.pred_dir)}.xlsx'))
        summarysheet = workbook.add_worksheet('summary')
        metricsheets = [workbook.add_worksheet(metric) for metric in self.results[0][2].keys()]
        
        for i, metric in enumerate(self.results[0][2].keys()):
            summarysheet.write(i, 0, metric)
            summarysheet.write(i, 1, f'={metric}!B2')
        
        for row, (dataset, clip, metrics) in enumerate(self.results):
            for metricsheet, metric in zip(metricsheets, metrics.values()):
                # Write the header
                if row == 0:
                    metricsheet.write(1, 0, 'Average')
                    metricsheet.write(1, 1, f'=AVERAGE(C2:ZZ2)')
                    for col in range(len(metric)):
                        metricsheet.write(0, col + 2, col)
                        colname = xlsxwriter.utility.xl_col_to_name(col + 2)
                        metricsheet.write(1, col + 2, f'=AVERAGE({colname}3:{colname}9999)')
                        
                metricsheet.write(row + 2, 0, dataset)
                metricsheet.write(row + 2, 1, clip)
                metricsheet.write_row(row + 2, 2, metric)
        
        workbook.close()

    def evaluate_worker(self, dataset, clip, position):
        framenames = sorted(os.listdir(os.path.join(self.args.pred_dir, dataset, clip, 'pha')))
        metrics = {metric_name : [] for metric_name in self.args.metrics}
        
        pred_pha_tm1 = None
        true_pha_tm1 = None
        
        for i, framename in enumerate(tqdm(framenames, desc=f'{dataset} {clip}', position=position, dynamic_ncols=True)):
            true_pha = cv2.imread(os.path.join(self.args.true_dir, dataset, clip, 'pha', framename), cv2.IMREAD_GRAYSCALE)
            pred_pha = cv2.imread(os.path.join(self.args.pred_dir, dataset, clip, 'pha', framename), cv2.IMREAD_GRAYSCALE)
            
            true_pha = torch.from_numpy(true_pha).cuda(non_blocking=True).float().div_(255)
            pred_pha = torch.from_numpy(pred_pha).cuda(non_blocking=True).float().div_(255)
            
            if 'pha_mad' in self.args.metrics:
                metrics['pha_mad'].append(self.mad(pred_pha, true_pha))
            if 'pha_mse' in self.args.metrics:
                metrics['pha_mse'].append(self.mse(pred_pha, true_pha))
            if 'pha_grad' in self.args.metrics:
                metrics['pha_grad'].append(self.grad(pred_pha, true_pha))
            if 'pha_conn' in self.args.metrics:
                metrics['pha_conn'].append(self.conn(pred_pha, true_pha))
            if 'pha_dtssd' in self.args.metrics:
                if i == 0:
                    metrics['pha_dtssd'].append(0)
                else:
                    metrics['pha_dtssd'].append(self.dtssd(pred_pha, pred_pha_tm1, true_pha, true_pha_tm1))
                    
            pred_pha_tm1 = pred_pha
            true_pha_tm1 = true_pha
            
            if 'fgr_mse' in self.args.metrics:
                true_fgr = cv2.imread(os.path.join(self.args.true_dir, dataset, clip, 'fgr', framename), cv2.IMREAD_COLOR)
                pred_fgr = cv2.imread(os.path.join(self.args.pred_dir, dataset, clip, 'fgr', framename), cv2.IMREAD_COLOR)
                
                true_fgr = torch.from_numpy(true_fgr).float().div_(255)
                pred_fgr = torch.from_numpy(pred_fgr).float().div_(255)
                
                true_msk = true_pha > 0
                metrics['fgr_mse'].append(self.mse(pred_fgr[true_msk], true_fgr[true_msk]))

        return metrics


class MetricMAD:
    def __call__(self, pred, true):
        return (pred - true).abs_().mean() * 1e3


class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3


class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        self.filter_x = torch.from_numpy(self.filter_x).unsqueeze(0).cuda()
        self.filter_y = torch.from_numpy(self.filter_y).unsqueeze(0).cuda()
    
    def __call__(self, pred, true):
        true_grad = self.gauss_gradient(true)
        pred_grad = self.gauss_gradient(pred)
        return ((true_grad - pred_grad) ** 2).sum() / 1000
    
    def gauss_gradient(self, img):
        img_filtered_x = kornia.filters.filter2D(img[None, None, :, :], self.filter_x, border_type='replicate')[0, 0]
        img_filtered_y = kornia.filters.filter2D(img[None, None, :, :], self.filter_y, border_type='replicate')[0, 0]
        return (img_filtered_x**2 + img_filtered_y**2).sqrt()
    
    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y
        
    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma**2


class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = dtSSD.sum() / true_t.numel()
        dtSSD = dtSSD.sqrt()
        return dtSSD * 1e2


if __name__ == '__main__':
    Evaluator()