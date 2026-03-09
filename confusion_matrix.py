import torch
from dataset import class_names, test_loader
from model import load_model_params, model, device
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# 1) 在 test 函数里收集
y_trues, y_preds = [], []
load_model_params(model, "./checkpoints/model_best.pth")
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        y_preds.extend(preds.numpy())
        y_trues.extend(y.numpy())

if __name__ == "__main__":
    with open("confusion_matrix.txt", "w") as f:
        print(classification_report(y_trues, y_preds, target_names=class_names), file=f)

    cm = confusion_matrix(y_trues, y_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.gcf().set_size_inches(36, 32)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=400)
