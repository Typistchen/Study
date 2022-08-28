import csv
import random

import torch
import os, glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image


class Hand(Dataset):
    def __init__(self, root, resize, mode):
        super(Hand, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}  #
        for name in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        print(self.name2label)
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train': # 60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]
        # image label

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images_path = []
            for name in self.name2label.keys():
                # \\rps\\paper\\000001.png
                images_path += glob.glob(os.path.join(self.root, name, '*png'))
                images_path += glob.glob(os.path.join(self.root, name, '*jpg'))

            print(len(images_path), images_path)
            random.shuffle(images_path)

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img_path in images_path:
                    name = img_path.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img_path, label])

        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x - mean) / std
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):

        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label


def main():
    import visdom
    import time
    ziv = visdom.Visdom()

    db = Hand('D:\\WorkRoom\\postgraduate\\Study\\Day5\\rps', 224, 'train')
    x, y = next(iter(db))
    # print(x, y)
    print('sample:', x.shape, y.shape)
    ziv.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    loader = DataLoader(db, batch_size=32, shuffle=True)
    for x, y in loader:
        ziv.images(db.denormalize(x), nrow=8, win='sample_x', opts=dict(title='sample_x'))
        ziv.text(str(y.numpy()), win='labels', opts=dict(title='batch-y'))
        time.sleep(10)

    # 快捷API
    # tf = transforms.Compose([
    #                 transforms.Resize((64,64)),
    #                 transforms.ToTensor(),
    # ])
    # db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # print(db.class_to_idx)
    #
    # for x,y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)

if __name__ == '__main__':
    main()