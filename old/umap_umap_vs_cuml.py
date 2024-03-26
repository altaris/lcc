from datetime import datetime

from cuml import UMAP as cUMAP
from umap import UMAP

from nlnas.dataset import TorchvisionDataset
from nlnas.utils import dl_head

datamodule = TorchvisionDataset("mnist")
datamodule.setup("fit")
x, y = dl_head(datamodule.train_dataloader(), 5000)
x = x.flatten(1).numpy()

kwargs = {"n_neighbors": 15, "n_components": 2, "metric": "euclidean"}

start = datetime.now()
e = cUMAP(**kwargs).fit_transform(x)
print("CUML", datetime.now() - start, "returned", type(e))

start = datetime.now()
e = UMAP(**kwargs).fit_transform(x)
print("UMAP-LEARN", datetime.now() - start, "returned", type(e))
