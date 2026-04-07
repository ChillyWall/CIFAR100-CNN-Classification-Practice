import torchvision
import cv2
import os
import json
import numpy as np
from tqdm import tqdm


def export_cifar100(save_dir="./", num_to_export=10000):
    # 1. 初始化路径
    img_dir = os.path.join(save_dir, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # 2. 下载并加载原始测试数据 (transform=None 确保不转为 Tensor)
    print("正在读取数据集...")
    dataset = torchvision.datasets.CIFAR100(
        root="./dataset", train=False, download=True, transform=None
    )

    # 原始数据是 numpy.ndarray, 形状 (10000, 32, 32, 3)
    raw_data = dataset.data
    targets = dataset.targets
    classes = dataset.classes

    metadata = []

    print(f"开始导出前 {num_to_export} 张图片...")
    for i in tqdm(range(min(num_to_export, len(raw_data)))):
        # --- 获取原始图片 (RGB) ---
        img_rgb = raw_data[i]

        # --- 仅色彩空间转换：RGB 转 BGR (OpenCV 写入必备) ---
        # 这一步是为了保证图片颜色正常，不涉及数据缩放或归一化
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # --- 生成补零文件名 ---
        file_name = f"{i:02d}.png"  # 格式如 00001.png
        file_path = os.path.join(img_dir, file_name)

        # --- 使用 OpenCV 写入 ---
        cv2.imwrite(file_path, img_bgr)

        # --- 记录元数据 ---
        label_idx = targets[i]
        metadata.append(
            {
                "file_name": file_name,
                "class_index": label_idx,
                "class_name": classes[label_idx],
            }
        )

    # 3. 导出 JSON 文件
    json_path = os.path.join(save_dir, "metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\n导出完成！")
    print(f"图片目录: {img_dir}")
    print(f"配置文件: {json_path}")


if __name__ == "__main__":
    # 你可以修改 num_to_export 为 10000 来导出整个测试训练集
    export_cifar100(num_to_export=10000)
