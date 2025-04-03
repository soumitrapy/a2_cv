import os
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from scipy.io import loadmat
import numpy as np

class BSD500Dataset(Dataset):
    IMAGE_PATH, GT_PATH = 'IMAGE_PATH', 'GT_PATH'
    SEG_LABEL_TYPE = 'SEGMENT'
    BOUNDARY_LABEL_TYPE = 'BOUNDARY'
    LABEL_TYPES = [SEG_LABEL_TYPE, BOUNDARY_LABEL_TYPE]

    SPLITS = ['train', 'test', 'val']

    def __init__(self, root, split='train', label='BOUNDARY', transform=None, target_transform=None):
        '''

        :param root: root directory of the dataset (parent dir of the actual data)
        :param split: train/val/test
        :param label: either boundary or segmentation
        '''
        assert split in self.SPLITS
        assert label in self.LABEL_TYPES

        self.split = split
        self.root = root
        self.label_type = label
        self.transform = transform
        self.target_transform = target_transform

        self.images_dir = os.path.join(self.root, 'data', 'images', self.split)
        if not os.path.isdir(self.images_dir):  # in case you passed the base of the unzipped data as root
            self.root = f'{self.root}/BSR/BSDS500'
            self.images_dir = os.path.join(self.root, 'data', 'images', self.split)

        self.gt_dir = os.path.join(self.root, 'data', 'groundTruth', self.split)

        assert os.path.isdir(self.images_dir) and os.path.isdir(
            self.gt_dir), f'images or GTs images does not exist (imgs dir: {self.images_dir})'

        self.images = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)]
        self.gts = set([os.path.join(self.gt_dir, f) for f in os.listdir(self.gt_dir)])

        self.labeled_image_paths = []
        for im_path in sorted(self.images):
            im = os.path.basename(im_path).split('.')[0]
            gt_path = os.path.join(self.gt_dir, f'{im}.mat')
            if gt_path in self.gts:
                self.labeled_image_paths.append({self.IMAGE_PATH: im_path, self.GT_PATH: gt_path})

        self.num_samples = len(self.labeled_image_paths)

    def __getitem__(self, idx):
        paths = self.labeled_image_paths[idx]
        img_path = paths[self.IMAGE_PATH]
        gt_path = paths[self.GT_PATH]

        image = np.array(Image.open(img_path).convert('RGB'))
        raw_gt = loadmat(gt_path)
        gTruth = raw_gt['groundTruth'][0][0][0][0][self.LABEL_TYPES.index(self.label_type)]
        if image.shape==(481,321,3):
            image = image.reshape((321,481,3))
            gTruth = gTruth.reshape((321,481))
        assert gTruth.shape == (321,481)
        image = Image.fromarray(image)
        gTruth = Image.fromarray(gTruth)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gTruth = self.target_transform(gTruth)

        #item_data = {'image': image, 'gTruth':gTruth, 'im_path': img_path, 'gt_path': gt_path}
        return image, gTruth

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    dataset_root = './BSR/BSDS500'

    from torchvision import transforms
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])
    target_transform = transforms.ToTensor()
    ds = BSD500Dataset(root=dataset_root, split='test', label='BOUNDARY', transform=transform, target_transform=target_transform)
    dl = DataLoader(ds, batch_size=10)
    
    