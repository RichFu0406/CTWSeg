import numpy as np
import SharedArray as SA
import torch

from util.voxelize import voxelize


def sa_create(name, var):
    shared = SA.create(name, var.shape, dtype=var.dtype)
    shared[...] = var[...]
    shared.flags.writeable = False
    return shared


def collate_fn(batch):
    coord, feat, target = list(zip(*batch))
    offset = []
    count = 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)

    return torch.cat(coord), torch.cat(feat), torch.cat(target), torch.IntTensor(offset)


def data_prepare(
    coord,
    feat,
    label,
    split='train',
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord -= np.min(coord, 0)
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuffle_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuffle_idx)
        coord, feat, label = coord[shuffle_idx], feat[shuffle_idx], label[shuffle_idx]

    coord -= np.min(coord, 0)
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.0
    label = torch.LongTensor(label)
    return coord, feat, label
