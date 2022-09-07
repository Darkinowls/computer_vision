import os
import numpy as np

from opencv_eyes import get_image_edge


def get_edges_array(db):
    edges = []

    dirs = os.listdir(db)
    for directory in dirs:
        files = os.listdir('%s/%s' % (db, directory))
        for f in files:
            path = '%s/%s/%s' % (db, directory, f)
            edges.append(get_image_edge(path))
            print(path)

    edges = np.array(edges)
    return edges
