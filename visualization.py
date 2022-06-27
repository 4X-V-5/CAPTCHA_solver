import torch
import model

if __name__ == "__main__":
    model = model.ResNet(model.ResidualBlock)
    model.load_model(r"models/ResNet18_epoch45_lr0.001.pth")
    data = torch.rand(1, 1, 40, 120)
    torch.onnx.export(model, data, 'model.onnx', export_params=True, opset_version=11)