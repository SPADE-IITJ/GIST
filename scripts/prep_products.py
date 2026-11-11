"""Convert DGL's GraphBolt copy of ogbn-products (data.dgl.ai/dataset/graphbolt/ogbn-products.zip) into the object
gist's load_ogb returns: a PyG Data with x [N, 100] float32, y [N] long (OGB labels, squeezed), undirected
edge_index; plus in_dim and out_dim. Checked against OGB's published statistics before saving.

    python scripts/prep_products.py data/dl/gb/ogbn-products data/ogbn_products_pyg/data.pt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
x = torch.from_numpy(np.load(src / "data" / "node-feat.npy")).float()
y = torch.from_numpy(np.load(src / "data" / "node-label.npy")).long().reshape(-1)
e = pd.read_csv(src / "edges" / "bi_edge.csv", header=None, dtype=np.int64, engine="c").to_numpy()
ei = torch.from_numpy(e.T.copy())
n = x.shape[0]
ei = to_undirected(ei, num_nodes=n)
und = int((ei[0] <= ei[1]).sum())   # OGB counts its 128 self-loops
stats = dict(nodes=n, undirected_edges=und, features=int(x.shape[1]), classes=int(y.max()) + 1)
expect = dict(nodes=2449029, undirected_edges=61859140, features=100, classes=47)
print("got", stats, "\nOGB", expect)
assert stats == expect, "does not match OGB's published ogbn-products statistics"
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(dict(data=Data(x=x, y=y, edge_index=ei, num_nodes=n), in_dim=100, out_dim=47,
                source="DGL GraphBolt ogbn-products (data.dgl.ai/dataset/graphbolt/ogbn-products.zip)", stats=stats), dst)
print("wrote", dst)
