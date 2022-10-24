import numpy as np

def shape_unfold_transform(contour):
    coord = contour
    coord_mean = np.mean(coord, axis=0)
    dist = np.linalg.norm(coord-coord_mean, axis=1)