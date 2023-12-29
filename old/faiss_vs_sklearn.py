from datetime import datetime

import numpy as np
from faiss import IndexFlatL2
from sklearn.neighbors import NearestNeighbors

n, d, k = 2048, 2048, 5

x = np.random.rand(n, d)

print("SKLEARN")
start = datetime.now()
index = NearestNeighbors(n_neighbors=k)
index.fit(x)
print("Build index", datetime.now() - start)
start = datetime.now()
dst, idx = index.kneighbors(x)
print("Search index", datetime.now() - start)
print(dst[:3])
print(idx[:3])
print(np.linalg.norm(x[0] - x[idx[0, 1]]), dst[0, 1])


print()
print()
print("FAISS")
start = datetime.now()
index = IndexFlatL2(d)
index.add(x)
print("Build index", datetime.now() - start)
start = datetime.now()
dst, idx = index.search(x, k)
print("Search index", datetime.now() - start)
print(np.sqrt(dst[:3]))
print(idx[:3])
print(np.linalg.norm(x[0] - x[idx[0, 1]]), dst[0, 1])
