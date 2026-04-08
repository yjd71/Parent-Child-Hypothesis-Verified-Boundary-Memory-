import os
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils import preproc
from utils import path_to_image
from utils import Logger


Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning

logger = Logger(name='Dataset', path='./data/logs/dataset.txt')

class MyData(data.Dataset):
    def __init__(self, config, datasets, image_size, is_train=True):
        self.config = config
        self.size_train = image_size
        self.size_test = image_size
        self.keep_size = not config.img_size
        self.data_size = (config.img_size, config.img_size)
        self.is_train = is_train
        self.load_all = config.load_all
        # self.device = config.device
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ][self.load_all or self.keep_size:])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ][self.load_all or self.keep_size:])
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split('+'):
            image_root = os.path.join(dataset_root, dataset, 'im')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root)]
        self.label_paths = []
        for p in self.image_paths:
            for ext in ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace('/im/', '/gt/').replace('.'+p.split('.')[-1], ext)
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    break
        self.image_to_idx = {os.path.basename(image_path).split('.')[0]: idx for idx, image_path in enumerate(self.image_paths)}
        self.unlabeled_indices = []

        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=(config.img_size, config.img_size), color_type='rgb')
                _label = path_to_image(label_path, size=(config.img_size, config.img_size), color_type='gray')
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)

    def random_boolean(self, probability_of_true=0.5):
        return random.random() < probability_of_true
    
    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
        else:
            image = path_to_image(self.image_paths[index], size=(self.config.img_size, self.config.img_size), color_type='rgb')
            label = path_to_image(self.label_paths[index], size=(self.config.img_size, self.config.img_size), color_type='gray')
        # loading image and label
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=self.config.preproc_methods)

        image, label = self.transform_image(image), self.transform_label(label)

        #! Preventing the semi-supervised learning from using the labeled data
        # if index in self.unlabeled_indices:
        #     label = 0
        
        hash_label_path = self.label_paths[index].split('/')[-1].split('.')[0]
        
        if self.is_train:
            return image, label, hash_label_path, index
        else:
            return image, label, self.label_paths[index], hash_label_path
    
    def set_unlabeled_data(self, unlabeled_indices):
        self.unlabeled_indices = unlabeled_indices
    
    def unset_unlabled_data(self):
        self.unlabeled_indices = []
        
    def __len__(self):
        return len(self.image_paths)

def prepare_dataloader(dataset: data.Dataset, batch_size: int, num_workers: int, to_be_distributed: bool=False, is_train: bool=False, labeled_indices: list=None, is_unsup=False) -> data.DataLoader:
    if labeled_indices:
        indices = [dataset.image_to_idx[file_name] for file_name in labeled_indices]
        if is_unsup:
            indices = np.setdiff1d(np.arange(len(dataset)), indices).tolist()
            sample_number = len(labeled_indices)
        else:
            sample_number = len(dataset) - len(labeled_indices)
    else:
        indices = None
    if indices and len(indices) < sample_number and is_train:
        indices = indices * ((sample_number + len(indices) - 1) // len(indices))
        indices = random.sample(indices, sample_number)
    if indices is not None and is_unsup:
        dataset.set_unlabeled_data(indices)
    if to_be_distributed:
        if indices is None:
            sampler = data.DistributedSampler(dataset)
        else:
            dataset = data.Subset(dataset, indices)
            logger.key_info("[+] Subset of dataset has been created, length: {}".format(len(dataset)))
            sampler = data.DistributedSampler(dataset)
        return data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=sampler, drop_last=True
        )
    else:
        if indices is None or len(indices) == 0 :
            sampler = data.RandomSampler(dataset)
        else:
            sampler = data.SubsetRandomSampler(indices)
        return data.DataLoader(
            dataset = dataset, batch_size = batch_size, num_workers=min(num_workers, batch_size, 0), pin_memory=True,
            sampler = sampler, drop_last=True
        )

def init_trainloader(config):
    train_loader = prepare_dataloader(
        MyData(config=config, datasets=config.training_set, image_size=config.img_size, is_train=True),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        to_be_distributed=config.distributed_train,
        is_train=True 
    )
    logger.success_info("{} batches of train dataloader {} has been created.".format(len(train_loader), config.training_set))
    return train_loader

def init_testloaders(config):
    test_loaders = {}
    for testset in config.testing_sets.strip('+').split('+'):
        _data_loader_test = prepare_dataloader(
            MyData(config=config, datasets=testset, image_size=config.img_size, is_train=False),
            num_workers=config.num_workers,
            to_be_distributed=False,
            batch_size=config.batch_size_valid, is_train=False
        )
        logger.success_info("{} batches of test dataloader {} has been created.".format(len(_data_loader_test), testset))
        test_loaders[testset] = _data_loader_test
    return test_loaders
