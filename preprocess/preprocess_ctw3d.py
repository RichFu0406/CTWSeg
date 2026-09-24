#!/usr/bin/env python3
"""Convert CTW3D-Parts annotations into scene-level NumPy files."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


META_DIR = Path(__file__).resolve().parent / 'meta'


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess CTW3D-Parts for CTWSeg.')
    parser.add_argument(
        '-i',
        '--data-root',
        type=Path,
        required=True,
        help='Path to the CTW3D-Parts directory.',
    )
    parser.add_argument(
        '-o',
        '--output-root',
        type=Path,
        required=True,
        help='Directory used to save room-level .npy files.',
    )
    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=min(8, os.cpu_count() or 1),
        help='Number of rooms processed in parallel.',
    )
    return parser.parse_args()


def read_meta(filename):
    path = META_DIR / filename
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def collect_room(annotation_dir, class_to_label, class_prefixes):
    """Merge all object files in one room into an XYZRGB-label array."""
    annotation_files = sorted(annotation_dir.glob('*.txt'))
    if not annotation_files:
        raise FileNotFoundError(f'No annotation files found in {annotation_dir}')

    blocks = []
    for annotation_file in annotation_files:
        stem = annotation_file.stem
        class_name = next(
            (
                name
                for name in class_prefixes
                if stem == name or stem.startswith(name + '_')
            ),
            None,
        )
        if class_name is None:
            raise ValueError(
                f'{annotation_file.name}: class is not listed in class_names.txt'
            )

        points = np.fromfile(annotation_file, dtype=np.float64, sep=' ')
        points = points.reshape(-1, 6)
        blocks.append((points, class_to_label[class_name]))

    room = np.empty((sum(points.shape[0] for points, _ in blocks), 7))
    start = 0
    for points, label in blocks:
        end = start + points.shape[0]
        room[start:end, :6] = points
        room[start:end, 6] = label
        start = end
    # room[:, :3] -= room[:, :3].min(axis=0)
    return room


def convert_room(task):
    (
        data_root,
        output_root,
        relative_path,
        class_to_label,
        class_prefixes,
    ) = task
    annotation_dir = data_root / relative_path
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f'Room path not found: {annotation_dir}')

    area_name = annotation_dir.parent.parent.name
    room_name = annotation_dir.parent.name
    output_path = output_root / f'{area_name}_{room_name}.npy'

    room = collect_room(
        annotation_dir,
        class_to_label,
        class_prefixes,
    )
    np.save(output_path, room)
    return output_path.name, room.shape[0]


def print_progress(results, total):
    for converted, (filename, point_count) in enumerate(results, 1):
        print(f'[{converted}/{total}] {filename}: {point_count} points')


def preprocess(data_root, output_root, workers):
    data_root = data_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    class_names = read_meta('class_names.txt')
    annotation_paths = read_meta('anno_paths.txt')
    class_to_label = {name: index for index, name in enumerate(class_names)}
    class_prefixes = sorted(class_names, key=len, reverse=True)

    print(f'Rooms from anno_paths.txt: {len(annotation_paths)}')
    print(
        'Labels from class_names.txt: '
        + ', '.join(
            f'{label}:{name}' for name, label in class_to_label.items()
        )
    )

    tasks = [
        (
            data_root,
            output_root,
            relative_path,
            class_to_label,
            class_prefixes,
        )
        for relative_path in annotation_paths
    ]
    workers = min(workers, len(tasks))
    if workers == 1:
        print_progress(map(convert_room, tasks), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            print_progress(executor.map(convert_room, tasks), len(tasks))

    print(f'Done. converted={len(tasks)}, workers={workers}, output_root={output_root}')


def main():
    args = parse_args()
    preprocess(args.data_root, args.output_root, args.workers)


if __name__ == '__main__':
    main()
