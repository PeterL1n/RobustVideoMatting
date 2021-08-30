"""
LR (Low-Resolution) evaluation.

Note, the script only does evaluation. You will need to first inference yourself and save the results to disk
Expected directory format for both prediction and ground-truth is:

    videomatte_512x288
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
    --pred-dir PATH_TO_PREDICTIONS/videomatte_512x288 \
    --true-dir PATH_TO_GROUNDTURTH/videomatte_512x288
    
An excel sheet with evaluation results will be written to "PATH_TO_PREDICTIONS/videomatte_512x288/videomatte_512x288.xlsx"
"""


import argparse
import os
import cv2
import numpy as np
import xlsxwriter
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
            'pha_mad', 'pha_mse', 'pha_grad', 'pha_conn', 'pha_dtssd', 'fgr_mad', 'fgr_mse'])
        self.args = parser.parse_args()
        
    def init_metrics(self):
        self.mad = MetricMAD()
        self.mse = MetricMSE()
        self.grad = MetricGRAD()
        self.conn = MetricCONN()
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
            true_pha = cv2.imread(os.path.join(self.args.true_dir, dataset, clip, 'pha', framename), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            pred_pha = cv2.imread(os.path.join(self.args.pred_dir, dataset, clip, 'pha', framename), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
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
            
            if 'fgr_mse' in self.args.metrics or 'fgr_mad' in self.args.metrics:
                true_fgr = cv2.imread(os.path.join(self.args.true_dir, dataset, clip, 'fgr', framename), cv2.IMREAD_COLOR).astype(np.float32) / 255
                pred_fgr = cv2.imread(os.path.join(self.args.pred_dir, dataset, clip, 'fgr', framename), cv2.IMREAD_COLOR).astype(np.float32) / 255
                true_msk = true_pha > 0
                
                if 'fgr_mse' in self.args.metrics:
                    metrics['fgr_mse'].append(self.mse(pred_fgr[true_msk], true_fgr[true_msk]))
                if 'fgr_mad' in self.args.metrics:
                    metrics['fgr_mad'].append(self.mad(pred_fgr[true_msk], true_fgr[true_msk]))

        return metrics

    
class MetricMAD:
    def __call__(self, pred, true):
        return np.abs(pred - true).mean() * 1e3


class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3


class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
    
    def __call__(self, pred, true):
        pred_normed = np.zeros_like(pred)
        true_normed = np.zeros_like(true)
        cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(true, true_normed, 1., 0., cv2.NORM_MINMAX)

        true_grad = self.gauss_gradient(true_normed).astype(np.float32)
        pred_grad = self.gauss_gradient(pred_normed).astype(np.float32)

        grad_loss = ((true_grad - pred_grad) ** 2).sum()
        return grad_loss / 1000
    
    def gauss_gradient(self, img):
        img_filtered_x = cv2.filter2D(img, -1, self.filter_x, borderType=cv2.BORDER_REPLICATE)
        img_filtered_y = cv2.filter2D(img, -1, self.filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
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


class MetricCONN:
    def __call__(self, pred, true):
        step=0.1
        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(true)
        for i in range(1, len(thresh_steps)):
            true_thresh = true >= thresh_steps[i]
            pred_thresh = pred >= thresh_steps[i]
            intersection = (true_thresh & pred_thresh).astype(np.uint8)

            # connected components
            _, output, stats, _ = cv2.connectedComponentsWithStats(
                intersection, connectivity=4)
            # start from 1 in dim 0 to exclude background
            size = stats[1:, -1]

            # largest connected component of the intersection
            omega = np.zeros_like(true)
            if len(size) != 0:
                max_id = np.argmax(size)
                # plus one to include background
                omega[output == max_id + 1] = 1

            mask = (round_down_map == -1) & (omega == 0)
            round_down_map[mask] = thresh_steps[i - 1]
        round_down_map[round_down_map == -1] = 1

        true_diff = true - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        true_phi = 1 - true_diff * (true_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

        connectivity_error = np.sum(np.abs(true_phi - pred_phi))
        return connectivity_error / 1000


class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = np.sum(dtSSD) / true_t.size
        dtSSD = np.sqrt(dtSSD)
        return dtSSD * 1e2



if __name__ == '__main__':
    Evaluator()