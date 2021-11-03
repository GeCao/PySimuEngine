# 导入枚举类
from enum import Enum
import numpy as np
import torch.nn.functional as F
import torch



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

A = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]).to(torch.float32).to("cpu")
print(A[::3, :])
print(F.pad(A[..., 1:, :].unsqueeze(0), pad=(0, 0, 0, 1), mode='constant').squeeze(0).view(-1))

print(A.sum(dim=0))

model_mat = np.array([1,1,1])
print(model_mat.shape)

shader_source = "in vec2 Temp_vec \n #include \"PCF.glsl\"  \n \n void main() \n{}  "
if "#include" in shader_source:
    shader_source_split = shader_source.split("#include ")
    print(shader_source_split)
    for i in range(len(shader_source_split)):
        if i > 0:
            split_idx = (shader_source_split[1].find('\"', 1))
            path_file = shader_source_split[1][1:split_idx]
            left_str = shader_source_split[1][split_idx + 1:]
            print(path_file)
            print(left_str)


"""
import numpy as np
np.random.seed(0)
a = np.linspace(start=0, stop=4, num=5).repeat(5).reshape(5, 5)
b = np.linspace(start=0, stop=4, num=5).repeat(5).reshape(5, 5).T
# print(a)
# print(b)

import numpy as np
from matplotlib.pyplot import plot, show
x = np.linspace(0, 2 * np.pi, 30) #创建一个包含30个点的余弦波信号
wave = np.cos(x)
transformed = np.fft.fft(wave)  #使用fft函数对余弦波信号进行傅里叶变换。
print(wave.shape)
print(transformed.shape)
print(np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9))  #对变换后的结果应用ifft函数，应该可以近似地还原初始信号。
plot(transformed)  #使用Matplotlib绘制变换后的信号。
show()
"""

