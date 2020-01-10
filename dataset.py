import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree

def analyze_xml(file_path):
    meta = xml.etree.ElementTree.parse(file_path).getroot()
    size = meta.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    obj = meta.find('object')
    box = obj.find('bndbox')
    xmin = int(box.find('xmin').text)
    ymin = int(box.find('ymin').text)
    xmax = int(box.find('xmax').text)
    ymax = int(box.find('ymax').text)

    x = (xmin + xmax) / 2 / img_width
    y = (ymin + ymax) / 2 / img_height
    bb_width = (xmax - xmin) / img_width
    bb_height = (ymax - ymin) / img_height

    return np.array([x, y, bb_width, bb_height], dtype='float32')
    # return np.array([x, y], dtype='float32')
    # return np.array([0, 0, 0, 0], dtype='float32')

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__()
        
        self.imgs = []
        self.xml = []
        self.transform = transform
        for dir in os.listdir(root):
            dir_path = os.path.join(root, dir)
            if os.path.isdir(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    file_suf = file_path.split('.')[-1]
                    if  file_suf == 'jpg':
                        self.imgs.append(file_path)
                    elif file_suf == 'xml':
                        self.xml.append(file_path)
                        # label = analyze_xml(file_path)
                        # self.labels.append(label)
        
        assert len(self.imgs) == len(self.xml), 'imgs lens most eq labels lens'
                        
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        label = analyze_xml(self.xml[index])
        label = torch.from_numpy(label)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

        