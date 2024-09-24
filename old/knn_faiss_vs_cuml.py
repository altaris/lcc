from datetime import datetime

import faiss
import numpy as np
import torch
from cuml.neighbors import NearestNeighbors

n, d, k = 10000, 4096, 5

x = np.random.random((n, d))

print()
print()
print("FAISS")
print()
print()
start = datetime.now()
index = faiss.IndexFlatL2(d)
index.add(x)
print("Add vectors", datetime.now() - start)
start = datetime.now()
dst, idx = index.search(x, k)
print("Query index", datetime.now() - start)
print(dst[:3])
print(idx[:3])
print(np.linalg.norm(x[0] - x[idx[0, 1]]))
print(dst[0, 1])

print()
print()
print("CUML CPU")
print()
print()
start = datetime.now()
model_cpu = NearestNeighbors(n_neighbors=k)
model_cpu.fit(x)
print("Train model", datetime.now() - start)
start = datetime.now()
dst, idx = model_cpu.kneighbors(x)
print("Search model", datetime.now() - start)
print(dst[:3])
print(idx[:3])
print(np.linalg.norm(x[0] - x[idx[0, 1]]))
print(dst[0, 1])
