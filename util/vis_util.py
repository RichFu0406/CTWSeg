import numpy as np


CTW3D_COLORS = [
    [0, 255, 0],
    [0, 0, 255],
    [0, 255, 255],
    [255, 255, 0],
    [255, 0, 255],
    [100, 100, 255],
    [200, 200, 100],
    [170, 120, 200],
    [255, 0, 0],
    [200, 100, 100],
    [10, 200, 100],
    [200, 200, 200],
    [50, 50, 50],
]


def write_ply_color(points, labels, out_filename, num_classes=None):
    """Write XYZ points with CTW3D-Parts label colors as an OBJ file."""
    labels = labels.astype(int)
    if num_classes is not None:
        assert num_classes > np.max(labels)

    with open(out_filename, 'w') as fout:
        for point, label in zip(points, labels):
            color = CTW3D_COLORS[label]
            fout.write(
                'v %f %f %f %d %d %d\n'
                % (point[0], point[1], point[2], color[0], color[1], color[2])
            )


def write_ply_rgb(points, rgb, out_filename, num_classes=None):
    """Write XYZ points with their input RGB colors as an OBJ file."""
    with open(out_filename, 'w') as fout:
        for point, color in zip(points, rgb):
            fout.write(
                'v %f %f %f %d %d %d\n'
                % (point[0], point[1], point[2], color[0], color[1], color[2])
            )
