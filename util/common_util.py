import os

import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def labeled_result_path(result_root, labeled_point, folder_name):
    """Build results/<label percentage>/<folder name>."""
    if not isinstance(labeled_point, str) or not labeled_point.endswith('%'):
        raise ValueError("labeled_point must be a percentage such as '0.1%'.")

    try:
        percentage = float(labeled_point[:-1])
    except ValueError as error:
        raise ValueError(
            "labeled_point must be a percentage such as '0.1%'."
        ) from error

    if not 0 < percentage <= 100:
        raise ValueError('labeled_point must be greater than 0% and at most 100%.')

    percentage_folder = format(percentage, 'g')
    return os.path.join(result_root, percentage_folder, folder_name)


def create_labeled_result_dirs(result_root, labeled_point):
    checkpoint_dir = labeled_result_path(result_root, labeled_point, 'ckpt')
    visualization_dir = labeled_result_path(result_root, labeled_point, 'vis')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    return checkpoint_dir, visualization_dir
