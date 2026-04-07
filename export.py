import torch
from model import CIFAR100_VGG, load_model_params


def export_onnx(model, checkpoint):
    model = CIFAR100_VGG()
    load_model_params(model, checkpoint)
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        "cifar100_vgg.onnx",
        verbose=True,
        input_names=["input"],  # 输入节点名称
        output_names=["output"],  # 输出节点名称
        opset_version=13,  # 算子集版本
        dynamic_axes={
            "input": {0: "batch_size"},  # 第0维 = 动态 batch
            "output": {0: "batch_size"},  # 输出第0维也跟着动态
        },
    )


if __name__ == "__main__":
    model = CIFAR100_VGG()
    export_onnx(model, "./checkpoints/model_best.pth")
