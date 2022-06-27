import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms as T
import preprocessing

torch.manual_seed(3407)

train_folder = r"data/augtrain_processed"
test_folder = r"data/test"


def label_to_num(label):
    """
    Separate each character and represent it with digits
    :param label: label with 4 characters
    :return: array with 4 digits
    """
    num = []
    for i in range(0, len(label)):
        if '0' <= label[i] <= '9':
            num.append(ord(label[i]) - ord('0'))
        elif 'a' <= label[i] <= 'z':
            num.append(ord(label[i]) - ord('a') + 10)
        else:
            num.append(ord(label[i]) - ord('A') + 36)
    return num


def num_to_label(numbers):
    """
    Transform array into a string with 4 characters
    :param numbers: array with 4 digits
    :return: string with 4 characters
    """
    label = ""
    for i in numbers:
        if i <= 9:
            label += chr(ord('0') + i)
        elif i <= 35:
            label += chr(ord('a') + i - 10)
        else:
            label += chr(ord('A') + i - 36)
    return label


class CAPTCHA(torch.utils.data.Dataset):
    """
    Create custom train set and validation set with preprocessed images
    """
    def __init__(self, image_folder) -> None:
        self.imgs_path = [os.path.join(image_folder, x) for x in os.listdir(image_folder)]
        self.transform = T.Compose([T.Resize((40, 120)),
                                    T.ToTensor(),
                                    T.Normalize((0.8569146394729614,), (0.18031486868858337,))])

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        label = torch.Tensor(label_to_num(img_path[42:46]))
        data = Image.open(img_path)
        data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.imgs_path)


class Test(torch.utils.data.Dataset):
    """
    Create custom test set with original images
    """
    def __init__(self, image_folder) -> None:
        self.imgs_path = [os.path.join(image_folder, x) for x in os.listdir(image_folder)]
        self.transform = T.Compose([T.Resize((40, 120)),
                                    T.ToTensor(),
                                    T.Normalize((0.8569146394729614,), (0.18031486868858337,))])

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        label = torch.Tensor(label_to_num(img_path[10:14]))
        data = cv2.imread(img_path)
        data = preprocessing.preprocessing(data)
        data = Image.fromarray(data)
        data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.imgs_path)


if __name__ == "__main__":
    dataset = CAPTCHA(train_folder)
    image, label = dataset[0]
    print(num_to_label(label), image.size)
    plt.imshow(image, cmap='gray')
    plt.show()
    # print(label_to_num('0123456789abcd'))
