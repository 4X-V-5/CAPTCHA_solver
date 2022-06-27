import numpy as np
from params import *
import torch
from torch import nn
from model import *
import torch.optim
import os
import my_dataset
import cv2
import tensorboardX
import torchvision.transforms as T
import torchvision
import PIL

torch.manual_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = r"models"
train_path = r"data/augtrain_processed"


def evaluate(model_path, loader, record_fail=False):
    model = ResNet(ResidualBlock)
    model.load_model(model_path)
    model = model.cuda(3)
    model.eval()
    loss = nn.CrossEntropyLoss()
    val_loss = 0
    total = len(loader.dataset)
    correct = 0
    for val_batch in loader:
        with torch.cuda.device('cuda:3'):
            torch.cuda.empty_cache()
        val_img, val_label = val_batch
        if torch.cuda.is_available():
            val_img = val_img.cuda(3)
            val_label = val_label.cuda(3)
        val_label = val_label.long()

        real_label = my_dataset.num_to_label([val_label[0][0], val_label[0][1], val_label[0][2], val_label[0][3]])

        val_label1, val_label2 = val_label[:, 0], val_label[:, 1]
        val_label3, val_label4 = val_label[:, 2], val_label[:, 3]

        val_out1, val_out2, val_out3, val_out4 = model(val_img)

        val_loss1, val_loss2 = loss(val_out1, val_label1), loss(val_out2, val_label2)
        val_loss3, val_loss4 = loss(val_out3, val_label3), loss(val_out4, val_label4)

        val_loss += val_loss1 + val_loss2 + val_loss3 + val_loss4

        val_out1, val_out2 = val_out1.topk(1, dim=1)[1].view(1, 1), val_out2.topk(1, dim=1)[1].view(1, 1)
        val_out3, val_out4 = val_out3.topk(1, dim=1)[1].view(1, 1), val_out4.topk(1, dim=1)[1].view(1, 1)
        val_out = torch.cat((val_out1, val_out2, val_out3, val_out4), dim=1)

        pred_label = my_dataset.num_to_label([val_out[0][0], val_out[0][1], val_out[0][2], val_out[0][3]])
        print("ground_truth: {}, predict: {}".format(real_label, pred_label))
        if real_label == pred_label:
            correct += 1
        else:
            if record_fail:
                torchvision.utils.save_image(val_img[0].mul_(0.18031486868858337).add_(0.8569146394729614), f'test_error/{real_label}_{pred_label}.jpg')
                # temp = T.ToPILImage(val_img[0].cpu())
                # # temp.save(f'test_error/{real_label}_{pred_label}.jpg')
                # cv2.imwrite(f'test_error/{real_label}_{pred_label}.jpg', np.asarray(temp, dtype=np.uint8))

    val_loss /= len(loader.dataset)
    accuracy = correct / total
    return val_loss, accuracy


def train(train_loader):
    model = ResNet(ResidualBlock)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    if torch.cuda.is_available():
        model = model.cuda(1)

    lowest_val_loss = np.inf

    for i in range(epoch):
        print("epoch: {}".format(i + 1))
        for train_batch in train_loader:
            with torch.cuda.device('cuda:1'):
                torch.cuda.empty_cache()
            img, label = train_batch
            if torch.cuda.is_available():
                img = img.cuda(1)
                label = label.cuda(1)
            label = label.long()
            label1, label2 = label[:, 0], label[:, 1]
            label3, label4 = label[:, 2], label[:, 3]

            out1, out2, out3, out4 = model(img)

            loss1, loss2 = loss(out1, label1), loss(out2, label2)
            loss3, loss4 = loss(out3, label3), loss(out4, label4)

            total_loss = loss1 + loss2 + loss3 + loss4

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(model_save_path, f'ResNet18_epoch{i+1}_lr{learning_rate}.pth'))


if __name__ == "__main__":
    # writer = tensorboardX.Summ
    dataset = my_dataset.CAPTCHA(train_path)
    train_set, val_set = torch.utils.data.random_split(dataset, [7503, 501])
    # train_loader = torch.utils.data.DataLoader(dataset=train_set,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            num_workers=2,
    #                                            pin_memory=False)
    #
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=2,
                                             pin_memory=False)

    train(train_loader)
    # models_path = r"models"
    # model_list = os.listdir(models_path)
    # for i in model_list:
    #     val_loss, accuracy = evaluate(os.path.join(models_path, i), val_loader)
    #     print("model {}:\n\tval_loss: {}, accuracy: {}%".format(i, val_loss, accuracy*100))

    model_name = r'models/ResNet18_epoch45_lr0.0002.pth'
    with torch.no_grad():
        val_loss, accuracy = evaluate(model_name, val_loader)
        print("{}:\n\tval_loss: {}, accuracy: {}%".format(model_name, val_loss, accuracy*100))
