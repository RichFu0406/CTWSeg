import os

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import data_prepare, sa_create


class CTW3DParts(Dataset):
    def __init__(
        self,
        split='train',
        data_root='trainval',
        test_area=5,
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        labeled_point='0.1%',
    ):
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop
        self.labeled_point = labeled_point

        room_names = sorted(
            filename[:-4]
            for filename in os.listdir(data_root)
            if filename.startswith('Area_') and filename.endswith('.npy')
        )
        area_prefix = f'Area_{test_area}_'
        if split == 'train':
            self.data_list = [
                name for name in room_names if not name.startswith(area_prefix)
            ]
        else:
            self.data_list = [
                name for name in room_names if name.startswith(area_prefix)
            ]

        if not self.data_list:
            raise RuntimeError(
                f'No {split} scenes found in {data_root!r}; run the CTW3D-Parts '
                'preprocessing step and verify test_area.'
            )

        for room_name in self.data_list:
            shared_name = f'shm://{room_name}'
            if not os.path.exists(f'/dev/shm/{room_name}'):
                room = np.load(os.path.join(data_root, room_name + '.npy'))
                sa_create(shared_name, room)

        self.data_idx = np.arange(len(self.data_list))
        print(f'Totally {len(self.data_idx)} samples in {split} set.')

    def __getitem__(self, index):
        room_index = self.data_idx[index % len(self.data_idx)]
        room = SA.attach(f'shm://{self.data_list[room_index]}').copy()

        if self.split in ('train', 'trainval'):
            ratio = float(self.labeled_point[:-1]) / 100.0
            labeled_count = max(int(room.shape[0] * ratio), 1)
            unlabeled_count = room.shape[0] - labeled_count
            unlabeled_indices = np.random.choice(
                room.shape[0],
                unlabeled_count,
                replace=False,
            )
            room[unlabeled_indices, 6] = 255

        coord, feat, label = room[:, :3], room[:, 3:6], room[:, 6]
        return data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
        )

    def __len__(self):
        return len(self.data_idx) * self.loop
