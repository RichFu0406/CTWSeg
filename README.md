# CTWSeg

CTWSeg is a PyTorch implementation for weakly supervised semantic segmentation
of the CTW3D-Parts point-cloud dataset. It provides a complete pipeline for
converting the released annotations, training with sparse point labels, and
evaluating predictions with an Area-based split.

## Dataset

CTW3D-Parts contains 24 scenes grouped into six Areas. Each point has XYZ and
RGB values, and annotations use seven semantic classes:

| ID | Class |
|---:|---|
| 0 | slope |
| 1 | citygates |
| 2 | wall |
| 3 | promenade |
| 4 | crenellation |
| 5 | parapet |
| 6 | other |

The dataset is not distributed in this repository. CTW3D-Parts can be downloaded
from Baidu Netdisk:

- Dataset: **CTW3D-Parts**
- Download: [Baidu Netdisk / 百度网盘](https://pan.baidu.com/s/1A5E8chZOPFXzT2rSiahwoA?pwd=w225)
- Extraction code / 提取码: `w225`

After downloading and extracting the dataset, keep the following directory layout:

```text
CTW3D-Parts/
├── Area_1/
│   ├── 1/
│   │   ├── 1.txt
│   │   └── Annotations/
│   │       ├── wall_1.txt
│   │       └── ...
│   └── ...
├── Area_2/
└── ...
```

Every text file stores `x y z r g b`, one point per line. The class of an
annotation file is determined by its filename prefix. The canonical scene list
and label order are defined in `preprocess/meta/anno_paths.txt` and
`preprocess/meta/class_names.txt`.

## Installation

The reference environment uses Ubuntu 20.04, Python 3.7, PyTorch 1.10.1, and
CUDA 11.3. A CUDA-capable NVIDIA GPU and a system CUDA toolkit containing
`nvcc` are required.

```bash
python -m pip install \
  torch==1.10.1+cu113 \
  torchvision==0.11.2+cu113 \
  torchaudio==0.10.1 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html
python -m pip install -r requirements.txt

cd lib/pointops
python setup.py install
cd ../..
```

## Data preparation

Convert the raw annotations into one NumPy array per scene:

```bash
python preprocess/preprocess_ctw3d.py \
  --data-root /path/to/CTW3D-Parts \
  --output-root data/ctw3d_parts
```

The converter writes files such as `Area_1_1.npy`. Each array has seven
columns: `x y z r g b label`. Use `--workers N` to change the number of scenes
processed concurrently.

## Training

Review `config.yaml` before training. Important options include:

- `test_area`: Area held out for validation and testing;
- `labeled_point`: percentage of training points retaining ground-truth labels;
- `data_root`: directory containing the preprocessed `.npy` files;
- `train_gpu`: visible GPU index.

Start training with:

```bash
python train.py --config config.yaml
```

The default `1%` setting stores checkpoints and TensorBoard events in
`results/1/ckpt`. Change configuration values from the command line by adding
key/value pairs after the config path, for example:

```bash
python train.py --config config.yaml TRAIN.labeled_point 0.1%
```

Training combines sparse-label cross entropy and Dice loss with an
exponential-moving-average teacher. High-confidence teacher predictions on
unlabeled points provide ramped pseudo-label supervision. The loss weights,
confidence threshold, temperature, and ramp-up duration are configurable in
`config.yaml`.

## Evaluation

By default, evaluation loads
`results/<labeled_point>/ckpt/model_best.pth` and writes predictions and OBJ
visualizations to `results/<labeled_point>/vis`:

```bash
python test.py --config config.yaml
```

Set `TEST.model_path` to evaluate a different checkpoint. Reported metrics are
class IoU, class accuracy, mean IoU, mean accuracy, and overall accuracy.

## Repository layout

```text
CTWSeg/
├── config.yaml                    # experiment configuration
├── model/ctwseg.py                # network definition
├── preprocess/preprocess_ctw3d.py # dataset converter
├── preprocess/meta/               # scene list and class mapping
├── util/                           # data, loss, metric, and visualization code
├── lib/pointops/                   # CUDA point-cloud operators
├── train.py
└── test.py
```

Raw data, generated NumPy arrays, checkpoints, logs, and visualizations are
excluded from version control.

## License

This repository is released under the MIT License. See `LICENSE` for the
required copyright and permission notice. Third-party components retain their
respective authorship and license terms.
