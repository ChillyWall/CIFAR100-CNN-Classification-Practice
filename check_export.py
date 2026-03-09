import onnx


def check_onnx_model(model_path: str):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型验证通过！")


if __name__ == "__main__":
    check_onnx_model("cifar100_vgg.onnx")
