# 导入枚举类
from enum import Enum
import numpy as np
import torch.nn.functional as F
import torch


'''
# 继承枚举类
class color(Enum):
    YELLOW = np.int
    BEOWN = np.float32
    LLM = np.float64
    RED = 2
    GREEN = 3
    PINK = 4

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[1, 1],
              [3, 4]])

A = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(torch.float32).to("cuda")
print(A)
print(F.pad(A[..., 1:, :].unsqueeze(0), pad=(0, 0, 0, 1), mode='constant').squeeze(0).view(-1))

print(A.sum(dim=0))

model_mat = np.array([1,1,1])
print(model_mat.shape)
'''






