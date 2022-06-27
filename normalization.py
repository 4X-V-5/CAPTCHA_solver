import numpy as np
import os
import cv2
import torch.utils.data

from my_dataset import CAPTCHA


# mean = 0
# std = 0
#
imgs_path = r'data/augtrain_processed'
# img_path = os.listdir(imgs_path)
#
# for i in img_path:
#     img = cv2.imread(os.path.join(imgs_path, i), cv2.IMREAD_GRAYSCALE)
#     mean += img[:, :].mean()
#     std += img[:, :].std()
#
# mean = mean / len(img_path)
# std = std / len(img_path)
#
train_set = CAPTCHA(imgs_path)
loader = torch.utils.data.DataLoader(train_set, batch_size=1000, num_workers=1)
num_of_pixels = len(train_set) * 120 * 40

total_sum = 0
sum_of_squared_error = 0
for batch in loader:
    total_sum += batch[0].sum()
mean = total_sum / num_of_pixels

for batch in loader:
    sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
std = torch.sqrt(sum_of_squared_error / num_of_pixels)

print("mean = {}, std = {}".format(mean, std))

