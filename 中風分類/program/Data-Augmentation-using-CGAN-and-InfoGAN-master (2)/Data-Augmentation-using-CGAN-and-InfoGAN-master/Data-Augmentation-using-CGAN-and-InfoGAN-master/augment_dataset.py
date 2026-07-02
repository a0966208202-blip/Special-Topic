from itertools import repeat
import numpy as np
import torch
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from build_dataset import DataLoader


class AugmentedDataloader:
    def get_augmented_dataloader(self, original_dataloader=DataLoader("stroke_ct"), split_percentage=1, batch_size=64,
                                 shuffle=True, num_workers=1, pin_memory=False,
                                 generated_dataset_path=None):
        # 如果沒有指定生成資料集路徑，則使用預設路徑（僅支援 stroke_ct）
        if generated_dataset_path is None:
            generated_dataset_path = './C_GAN_generate_datasets/2000k_image_set_stroke_ct_noise_1.pt'

        generated_image_tensor = torch.load(generated_dataset_path).detach()

        # 生成對應的標籤（兩類：0=Bleeding, 1=Ischemia）
        num_images_per_class = len(generated_image_tensor) // 2
        generated_labels = self.get_labels(num_images_per_class=num_images_per_class)

        subset_train_datasetA, subset_train_datasetB = \
            original_dataloader.get_train_dataset_split(split_percentage=split_percentage)
        subset_test_dataset = original_dataloader.get_test_dataset_split(split_percentage=split_percentage)

        original_data_tensor, original_label_tensor = self.get_original_image_and_label_tensors(subset_train_datasetB)

        augmented_data_tensor = torch.cat((original_data_tensor, generated_image_tensor), 0)
        augmented_label_tensor = torch.cat((original_label_tensor, generated_labels), 0)
        print("Original data tensor", original_data_tensor.size())
        print("Generated data tensor", generated_image_tensor.size())
        print("original_label_tensor", original_label_tensor.size())
        print("generated_labels", generated_labels.size())
        augmented_dataset = torch.utils.data.TensorDataset(augmented_data_tensor, augmented_label_tensor)
        print("augmented train dataset = ", len(augmented_dataset))
        print("augmented test dataset = ", len(subset_test_dataset))
        augmented_train_dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=batch_size,
                                                                 shuffle=shuffle, num_workers=num_workers,
                                                                 pin_memory=pin_memory)

        augmented_test_data_loader = torch.utils.data.DataLoader(
            subset_test_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)

        return augmented_train_dataloader, augmented_test_data_loader

    def get_original_image_and_label_tensors(self, dataset):
        global data_tensor, label_tensor
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        for i, (data, label) in enumerate(dataloader, 0):
            data_tensor = data
            label_tensor = label

        return data_tensor, label_tensor

    def get_labels(self, num_images_per_class=5400):
        """
        產生 CT 影像用的標籤：0 = Bleeding, 1 = Ischemia
        """
        # CT影像只有2個類別：Bleeding (0) 和 Ischemia (1)
        arr_0 = list(repeat(0, num_images_per_class))  # Bleeding
        arr_1 = list(repeat(1, num_images_per_class))  # Ischemia
        labels_2D = torch.tensor(arr_0 + arr_1)
        
        return labels_2D

if __name__ == '__main__':
    # 測試CT影像資料集擴增
    print("Testing CT image dataset augmentation...")
    original_dataloader = DataLoader("stroke_ct")
    augmented_dataset = AugmentedDataloader()

    # 注意：需要先訓練CGAN模型並生成影像才能使用此功能
    try:
        aug_train_dataloader, aug_test_dataloader = augmented_dataset.get_augmented_dataloader(
            original_dataloader=original_dataloader,
            split_percentage=0.9,
            generated_dataset_path='./C_GAN_generate_datasets/2000k_image_set_stroke_ct_noise_1.pt'
        )
        print("CT image augmentation successful!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the CGAN model first and generate images before using augmentation.")
