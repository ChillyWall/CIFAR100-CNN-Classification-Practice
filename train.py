import torch
from model import (
    device,
    scheduler,
    model,
    criterion,
    optimizer,
)
from dataset import train_loader, val_loader, test_loader
import csv
import numpy as np


def save_test_result(correct_rate: np.ndarray, mean, std, acc):
    with open("test_result.csv", "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        data = np.append(correct_rate, [mean, std, acc])
        writer.writerow(data)


def test(model, dataloader):
    model.eval()
    class_correct = np.zeros(100)
    class_total = np.zeros(100)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(images)):
                label = labels[i].item()  # 转换为 Python 整数
                class_correct[label] += c[i].item()
                class_total[label] += 1

    correct_rate = class_correct / class_total
    mean = np.mean(correct_rate)
    std = np.std(correct_rate)
    cv = std / mean if mean != 0 else 0
    acc = mean - cv * 0.7

    save_test_result(correct_rate, mean, std, acc)
    return mean


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    best_model_acc,
    epoch: int,
):
    model.train()
    decrease_times = 0
    history = {"train_loss": [], "val_acc": []}

    for i in range(epoch):
        running_loss = 0.0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.detach().item()

            if batch_idx % 100 == 0:
                avg_loss = running_loss / (batch_idx + 1)  # 累计平均损失
                print(
                    f"Train Epoch: {i} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tAvg Loss: {avg_loss:.6f}"
                )

        # 计算epoch总训练损失
        epoch_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(epoch_train_loss)
        print(f"Epoch {i} total loss: {epoch_train_loss:.6f}")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"current learning rate: {current_lr:.6f}")

        # 评估模型准确率并更新最佳模型参数文件
        model_acc = test(model, val_loader)
        if model_acc > best_model_acc:
            best_model_acc = model_acc
            torch.save(model.state_dict(), "./checkpoints/model_best.pth")
            print("Best model is updated whose acc is {}.".format(model_acc))
            decrease_times = 0
        else:
            print("The model acc is decreased to {}".format(model_acc))
            decrease_times += 1

        if (i + 1) % 5 == 0:
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            print("The train acc is {}\nThe test acc is {}".format(train_acc, test_acc))

        scheduler.step(model_acc)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            "./checkpoints/model_latest.pth",
        )
        print("The training status is saved.")

        if decrease_times >= 15:
            break

        print("\n")


if __name__ == "__main__":
    # load_model_params(model, "./checkpoints/model_best_v1.pth")
    best_model_acc = 0
    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        best_model_acc,
        200,
    )
