from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import csv
import numpy as np


def calculate_cifar100_stats(train_loader):
    # 初始化三个通道的像素值总和、平方和以及总像素数
    channel_sum = torch.tensor([0.0, 0.0, 0.0])
    channel_sq_sum = torch.tensor([0.0, 0.0, 0.0])
    num_pixels = 0

    for images, _ in train_loader:
        # 图像形状：[batch_size, channels, height, width]
        batch_size, channels, height, width = images.shape
        num_pixels += batch_size * height * width

        # 计算每个通道的总和
        channel_sum += images.sum(dim=[0, 2, 3])
        # 计算每个通道的平方和
        channel_sq_sum += (images**2).sum(dim=[0, 2, 3])

    # 计算均值
    mean = channel_sum / num_pixels
    # 计算标准差：std = sqrt(E[x²] - (E[x])²)
    std = (channel_sq_sum / num_pixels - mean**2) ** 0.5

    return mean, std


if __name__ == "__main__":
    train_dataset = datasets.CIFAR100(
        root="./dataset", train=True, download=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True, num_workers=16
    )
    mean, std = calculate_cifar100_stats(train_loader)
    with open("mean_std.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows([mean, std])
