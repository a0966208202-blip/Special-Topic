import torch
from torchvision import datasets, transforms
import os

class DataLoader:
    def __init__(self, dataset):
        self.dataset_name = dataset
        self.train_dataset = None
        self.test_dataset = None
        self.train_data_loader = None

    def get_train_dataloader(self, batch_size, shuffle=True, num_workers=1, pin_memory=False):
        self.__load_train_dataset()
        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)
        return self.train_data_loader

    def get_test_dataloader(self, batch_size, shuffle=False, num_workers=1, pin_memory=False):
        self.__load_test_dataset()
        test_data_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)
        return test_data_loader

    def __load_train_dataset(self):
        if self.dataset_name == "stroke_ct":
            self.__load_ct_scan_dataset(is_train=True, transform=self.__get_transform())
        elif self.dataset_name == "MNIST":
            self.__load_mnist_dataset(is_train=True, transform=self.__get_mnist_transform())
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")

    def __load_test_dataset(self):
        if self.dataset_name == "stroke_ct":
            self.__load_ct_scan_dataset(is_train=False, transform=self.__get_transform())
        elif self.dataset_name == "MNIST":
            self.__load_mnist_dataset(is_train=False, transform=self.__get_mnist_transform())
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")

    def __load_ct_scan_dataset(self, is_train=True, transform=None):
        """專門用來載入 CT 影像資料集的函數"""
        # 使用專案內的 train 資料夾
        base_path = os.path.dirname(os.path.abspath(__file__))
        if is_train:
            train_path = os.path.join(base_path, "train")
        else:
            train_path = os.path.join(base_path, "test")
            
        if not os.path.exists(train_path):
            raise ValueError(f"Path {train_path} does not exist!")
            
        if is_train:
            self.train_dataset = datasets.ImageFolder(
                root=train_path,
                transform=transform
            )
            print(f"Loaded {len(self.train_dataset)} training images from {train_path}")
            print(f"Found classes: {self.train_dataset.classes}")
        else:
            self.test_dataset = datasets.ImageFolder(
                root=train_path,
                transform=transform
            )
            print(f"Loaded {len(self.test_dataset)} test images from {train_path}")
            print(f"Found classes: {self.test_dataset.classes}")

    def __load_mnist_dataset(self, is_train=True, transform=None):
        """載入 MNIST 資料集"""
        if is_train:
            self.train_dataset = datasets.MNIST(
                root='./data/mnist',
                train=True,
                download=True,
                transform=transform
            )
            print(f"Loaded {len(self.train_dataset)} MNIST training images")
        else:
            self.test_dataset = datasets.MNIST(
                root='./data/mnist',
                train=False,
                download=True,
                transform=transform
            )
            print(f"Loaded {len(self.test_dataset)} MNIST test images")

    def __get_transform(self):
        """為 CT 影像設定的轉換流程"""
        # 灰階影像的標準正規化設定
        normalize = transforms.Normalize((0.5,), (0.5,))

        data_transform = transforms.Compose([
            transforms.Resize((128, 128)), # 將影像統一為 128x128
            transforms.Grayscale(num_output_channels=1), # 確保是單通道灰階
            transforms.ToTensor(),
            normalize,
        ])
        return data_transform

    def __get_mnist_transform(self):
        """為 MNIST 資料集設定的轉換流程"""
        normalize = transforms.Normalize((0.5,), (0.5,))
        
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        return data_transform

    def get_train_dataset_split(self, split_percentage=0.9):
        """將訓練資料集分割為兩部分"""
        if self.train_dataset is None:
            self.__load_train_dataset()
            
        total_size = len(self.train_dataset)
        split_size = int(total_size * split_percentage)
        
        subset_train_datasetA, subset_train_datasetB = torch.utils.data.random_split(
            self.train_dataset, [split_size, total_size - split_size]
        )
        
        return subset_train_datasetA, subset_train_datasetB

    def get_test_dataset_split(self, split_percentage=1.0):
        """獲取測試資料集"""
        if self.test_dataset is None:
            self.__load_test_dataset()
            
        if split_percentage == 1.0:
            return self.test_dataset
        else:
            total_size = len(self.test_dataset)
            split_size = int(total_size * split_percentage)
            subset_test_dataset, _ = torch.utils.data.random_split(
                self.test_dataset, [split_size, total_size - split_size]
            )
            return subset_test_dataset