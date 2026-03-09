from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import numpy as np
import torch


mean = (0.5071, 0.4865, 0.4409)
std = (0.2673, 0.2564, 0.2761)


class Cutout(object):
    """随机地在图像上遮挡一个方形区域"""

    def __init__(self, n_holes, length):
        # n_holes: 遮挡区域的数量 (通常为 1)
        # length: 遮挡区域的边长 (通常为 8 或 16)
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)  # H
        w = img.size(2)  # W

        mask = torch.ones((h, w), device=img.device, dtype=img.dtype)

        for n in range(self.n_holes):
            # 随机选择遮挡区域的中心点
            y = torch.randint(h, (1,))
            x = torch.randint(w, (1,))

            y1 = torch.clamp(y - self.length // 2, 0, h)
            y2 = torch.clamp(y + self.length // 2, 0, h)
            x1 = torch.clamp(x - self.length // 2, 0, w)
            x2 = torch.clamp(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        # 对图像的每个通道应用遮罩
        mask = mask.expand_as(img)
        img = img * mask

        return img


train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),  # 随机裁剪（带填充，增加边缘多样性）
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(1, 16),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

train_dataset = datasets.CIFAR100(
    root="./dataset", train=True, download=True, transform=train_transform
)
val_base_dataset = datasets.CIFAR100(
    root="./dataset", train=True, download=True, transform=test_transform
)
train_subset, val_subset = random_split(train_dataset, [40000, 10000])

val_indices = val_subset.indices
val_subset = Subset(val_base_dataset, val_indices)

test_dataset = datasets.CIFAR100(
    root="./dataset", train=False, transform=test_transform
)

train_loader = DataLoader(
    dataset=train_subset, batch_size=128, shuffle=True, num_workers=16
)
val_loader = DataLoader(
    dataset=val_subset, batch_size=128, shuffle=True, num_workers=16
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=128, shuffle=False, num_workers=16
)

# 展示 10 个样本
class_names = train_dataset.classes

denormalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],  # 反归一化均值：-mean/std
            std=[1 / s for s in std],  # 反归一化标准差：1/std
        )
    ]
)

if __name__ == "__main__":
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    for i in range(20):
        plt.figure(figsize=(12, 4))
        for j in range(5):
            # 反归一化后转换为numpy数组并调整通道顺序 (C, H, W) -> (H, W, C)
            img = denormalize(images[i * 5 + j]).numpy().transpose(1, 2, 0)
            # 确保像素值在 [0, 255] 范围内，并转换为uint8类型
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

            plt.subplot(1, 5, j + 1)
            plt.imshow(img)
            plt.title(class_names[labels[i * 5 + j].item()])
            plt.axis("off")

        plt.suptitle("CIFAR-100-{:02}".format(i + 1), fontsize=16)
        plt.tight_layout()
        plt.savefig("samples/cifar100_samples_{:02}.png".format(i + 1), dpi=300)
