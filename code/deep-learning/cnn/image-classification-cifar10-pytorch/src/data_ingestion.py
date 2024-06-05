import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


class CIFAR10DataIngestionPipeline:
    def __init__(self, root, batch_size=4, num_workers=2, preprocessor=None):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessor = preprocessor

    def get_train_dataloader(self):
        self.train_dataset = datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=self.preprocessor.preprocess if self.preprocessor else None,
        )
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_dataloader

    def get_test_dataloader(self):
        test_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=self.preprocessor.preprocess if self.preprocessor else None,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_dataloader

    def get_class_names(self):
        return self.train_dataset.classes
