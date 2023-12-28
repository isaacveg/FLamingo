"""
Output pictures of all clients to dataset_pictures/
"""

import os, sys
import numpy as np

base_dir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from core.utils import data_utils
from core.utils.data_utils import ClientDataset, save_data_to_picture
from PIL import Image

def merge_images(folder_path, output_path):
    images = []
    img_names = []
    image_width = 28  # 假设图片的宽度为 100 像素
    image_height = 28  # 假设图片的高度为 100 像素
    row_size = 10  # 每行放置的图片数量

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            images.append((filename, image))

    images.sort(key=lambda img: img[0])  # 按图片文件名排序
    images = [image[1] for image in images]  # 去掉文件名，只保留图片

    num_images = len(images)
    total_width = image_width * row_size
    # rows = (num_images + row_size - 1) // row_size
    rows = np.ceil(num_images / row_size).astype(int)
    total_height = image_height * rows

    merged_image = Image.new('L', (total_width, total_height), color=255)

    for i, image in enumerate(images):
        x_offset = (i % row_size) * image_width
        y_offset = (i // row_size) * image_height
        merged_image.paste(image, (x_offset, y_offset))

    merged_image.save(output_path)

# data_dir = '/data0/yzhu/FLMPI/datasets/'
# dataset_type = 'cifar10'
dataset_type = 'femnist'
data_dir = '/data0/yzhu/datasets/'
save_dir = '/data0/yzhu/FLMPI/analysis_tools/dataset_pictures/' + dataset_type
# for rank in range(1, 36):
#     folder_path = os.path.join(save_dir, str(rank))
#     output_path = os.path.join(save_dir, str(rank)+'_BIG.png')
#     dataset = ClientDataset(dataset_type, data_dir, rank=rank)
#     loader = dataset.get_train_loader(batch_size=32)
#     for batch_idx, data in enumerate(loader):
#         save_data_to_picture(dataset_type, data, folder_path)


for rank in range(1, 36):
    folder_path = os.path.join(save_dir, str(rank))
    output_path = os.path.join(save_dir, str(rank)+'_BIG.png')
    merge_images(folder_path, output_path)
