#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT影像CGAN訓練和資料擴增腳本
使用Conditional GAN來擴增Bleeding和Ischemia影像
"""

import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 匯入自定義模組
from build_dataset import DataLoader as CustomDataLoader
from CGAN_manager import CGANManager
from CONSTANTS import Constants
from augment_dataset import AugmentedDataloader

def main():
    """主函數：訓練CGAN模型並生成CT影像"""
    
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 設定資料集參數
    dataset_name = "stroke_ct"
    batch_size = Constants.C_GAN_BATCH_SIZE
    z_dim = Constants.CGAN_Z_DIM
    n_classes = Constants.CGAN_N_CLASSES
    
    print(f"資料集: {dataset_name}")
    print(f"批次大小: {batch_size}")
    print(f"雜訊維度: {z_dim}")
    print(f"類別數量: {n_classes}")
    
    # 載入資料
    print("\n載入CT影像資料...")
    try:
        dataloader = CustomDataLoader(dataset_name)
        train_dataloader = dataloader.get_train_dataloader(
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True if device.type == 'cuda' else False
        )
        print(f"成功載入訓練資料，共 {len(train_dataloader)} 個批次")
        
        # 顯示資料集資訊
        for images, labels in train_dataloader:
            print(f"影像形狀: {images.shape}")
            print(f"標籤形狀: {labels.shape}")
            print(f"類別: {torch.unique(labels)}")
            break
            
    except Exception as e:
        print(f"載入資料時發生錯誤: {e}")
        return
    
    # 建立CGAN管理器
    print("\n初始化CGAN管理器...")
    cgan_manager = CGANManager(
        device=device,
        dataset_name=dataset_name,
        dataloader=train_dataloader,
        n_classes=n_classes,
        z_dim=z_dim
    )
    
    # 訓練CGAN模型
    print("\n開始訓練CGAN模型...")
    print(f"訓練週期: {Constants.CGAN_EPOCH}")
    print(f"學習率: G={Constants.C_GAN_LR_G}, D={Constants.C_GAN_LR_D}")
    
    try:
        cgan_manager.train_CGAN()
        print("CGAN訓練完成！")
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        return
    
    # 生成影像
    print("\n開始生成CT影像...")
    try:
        # 生成每個類別1000張影像
        num_images_per_class = 1000
        generated_images, generated_labels = cgan_manager.generate_images(
            num_images_per_class=num_images_per_class,
            save_path='./C_GAN_generate_datasets'
        )
        
        print(f"成功生成 {len(generated_images)} 張影像")
        print(f"Bleeding類別: {(generated_labels == 0).sum().item()} 張")
        print(f"Ischemia類別: {(generated_labels == 1).sum().item()} 張")
        
    except Exception as e:
        print(f"生成影像時發生錯誤: {e}")
        return
    
    # 測試生成的影像
    print("\n測試生成的影像...")
    try:
        cgan_manager.test_CGAN()
        print("影像測試完成！")
    except Exception as e:
        print(f"測試影像時發生錯誤: {e}")
    
    # 測試資料擴增
    print("\n測試資料擴增功能...")
    try:
        augmented_dataloader = AugmentedDataloader()
        aug_train_dataloader, aug_test_dataloader = augmented_dataloader.get_augmented_dataloader(
            original_dataloader=dataloader,
            split_percentage=0.9,
            batch_size=batch_size,
            dataset_name=dataset_name
        )
        
        print(f"擴增後訓練資料: {len(aug_train_dataloader)} 個批次")
        print(f"測試資料: {len(aug_test_dataloader)} 個批次")
        print("資料擴增測試完成！")
        
    except Exception as e:
        print(f"資料擴增測試時發生錯誤: {e}")
    
    print("\n所有任務完成！")
    print("生成的文件:")
    print("- 訓練好的模型: checkpoint/C_GAN/C_GAN_FINAL_MODEL_stroke_ct.pt")
    print("- 生成的影像: C_GAN_generate_datasets/")
    print("- 生成的影像預覽: C_GAN_Images/Generated_CT_Images_stroke_ct.png")

def test_existing_model():
    """測試已存在的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "stroke_ct"
    
    # 檢查模型是否存在
    model_path = f'checkpoint/C_GAN/C_GAN_FINAL_MODEL_{dataset_name}.pt'
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("請先執行訓練程序")
        return
    
    print("載入已存在的模型...")
    
    # 載入資料
    dataloader = CustomDataLoader(dataset_name)
    train_dataloader = dataloader.get_train_dataloader(batch_size=32, shuffle=False)
    
    # 建立CGAN管理器
    cgan_manager = CGANManager(
        device=device,
        dataset_name=dataset_name,
        dataloader=train_dataloader,
        n_classes=2,
        z_dim=64
    )
    
    # 測試模型
    try:
        cgan_manager.test_CGAN()
        print("模型測試完成！")
    except Exception as e:
        print(f"測試模型時發生錯誤: {e}")

if __name__ == "__main__":
    print("CT影像CGAN訓練和資料擴增程式")
    print("=" * 50)
    
    # 檢查是否有已訓練的模型
    model_path = 'checkpoint/C_GAN/C_GAN_FINAL_MODEL_stroke_ct.pt'
    if os.path.exists(model_path):
        print("發現已存在的模型，是否要重新訓練？")
        print("1. 重新訓練 (y/Y)")
        print("2. 只測試現有模型 (n/N)")
        choice = input("請選擇 (y/n): ").lower()
        
        if choice == 'y':
            main()
        else:
            test_existing_model()
    else:
        print("未發現已存在的模型，開始訓練...")
        main()
