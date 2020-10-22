import numpy as np
import os.path 
import sys
import torch
import torch.nn.functional as F
from annoy import AnnoyIndex

def twod_map(array, mapping):
    new_array = [[mapping[j] for j in i] for i in array]
    return new_array

def create_index(X, index_type='annoy'):
    if index_type == 'faiss':

        X_cont = np.ascontiguousarray(X, dtype=np.float32)
        n, dim = X_cont.shape
        if n < 200000:
            index = faiss.IndexFlatIP(dim)

        else:
            n_cells = 2048
            index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(dim),
                dim,
                n_cells,
                faiss.METRIC_INNER_PRODUCT
            )

        index.nprobe = 16
        n_train = min(n, 1000000)
        index.train(X_cont[:n_train])
        index.add(X_cont)

    else: 
        n, dim = X.size()[0], X.size()[1]
        index = AnnoyIndex(dim, metric='angular')
        for i,v in enumerate(X):
            index.add_item(i, v)
        index.build(100)

    return index


def find_closest(embeddings, k, index, queries, index_type='annoy'):
    if index_type == 'faiss':
        points = embeddings[queries]
        # points = F.normalize(points)
        p = points.detach().numpy()
        q = np.ascontiguousarray(p, dtype=np.float32)
        neighbors = index.assign(q, k)

    else:
        neighbors = []
        for q in queries:
            vector = embeddings[q]
            nns = index.get_nns_by_vector(vector, k)
            neighbors.append(nns)

    return neighbors

