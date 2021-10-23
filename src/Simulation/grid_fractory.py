import torch


def Zero_Tensor(resolution, dtype=torch.float64, device="cuda"):
    """
    :param resolution: A list
    :param dtype:
    :param device:
    :return:
    """
    return torch.zeros(torch.Size(resolution)).to(dtype).to(device)

def Ones_Tensor(resolution, dtype=torch.float64, device="cuda"):
    """
    :param resolution: A list
    :param dtype:
    :param device:
    :return:
    """
    return torch.ones(torch.Size(resolution)).to(dtype).to(device)