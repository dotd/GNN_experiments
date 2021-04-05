import torch
import time
start_time = time.time()
print(f"{time.time() - start_time:.4f} start time")

from torch_geometric.data import Data

print(f"{time.time() - start_time:.4f} after tg time")

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)
print(f"{time.time() - start_time:.4f} end time")
