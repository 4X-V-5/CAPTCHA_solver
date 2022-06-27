import cv2
import torch
import my_dataset
from model import *
from train import evaluate

torch.manual_seed(3407)

if __name__ == "__main__":
    test_path = r"data/test"
    test_set = my_dataset.Test(test_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=1,
                                              shuffle=True,
                                              pin_memory=False)
    with torch.no_grad():
        _, accuracy = evaluate('models/ResNet18_epoch45_lr0.0001.pth', test_loader, record_fail=False)
        print(f"Accuracy on test set: {accuracy*100}%")
