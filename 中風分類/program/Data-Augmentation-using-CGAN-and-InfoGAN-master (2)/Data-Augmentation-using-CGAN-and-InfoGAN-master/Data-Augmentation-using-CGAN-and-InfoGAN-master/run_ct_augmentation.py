#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版CT影像CGAN訓練腳本
一鍵執行Bleeding和Ischemia影像擴增
"""

import sys
import os

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_cgan_ct import main, test_existing_model

def quick_start():
    """快速開始指南"""
    print("=" * 60)
    print("CT影像CGAN資料擴增程式")
    print("=" * 60)
    print()
    print("此程式將使用Conditional GAN來擴增以下類別的CT影像:")
    print("• Bleeding (出血)")
    print("• Ischemia (缺血)")
    print()
    print("請確保您的資料夾結構如下:")
    print("train/")
    print("├── Bleeding/")
    print("│   └── *.png (出血影像)")
    print("└── Ischemia/")
    print("    └── *.png (缺血影像)")
    print()
    
    # 檢查資料夾是否存在
    train_path = os.path.join(os.path.dirname(__file__), "train")
    bleeding_path = os.path.join(train_path, "Bleeding")
    ischemia_path = os.path.join(train_path, "Ischemia")
    
    if not os.path.exists(train_path):
        print("❌ 錯誤: 找不到 'train' 資料夾")
        print("請確保在正確的目錄下執行此程式")
        return False
        
    if not os.path.exists(bleeding_path):
        print("❌ 錯誤: 找不到 'train/Bleeding' 資料夾")
        return False
        
    if not os.path.exists(ischemia_path):
        print("❌ 錯誤: 找不到 'train/Ischemia' 資料夾")
        return False
    
    # 檢查影像數量
    bleeding_count = len([f for f in os.listdir(bleeding_path) if f.lower().endswith('.png')])
    ischemia_count = len([f for f in os.listdir(ischemia_path) if f.lower().endswith('.png')])
    
    print(f"✅ 找到 Bleeding 影像: {bleeding_count} 張")
    print(f"✅ 找到 Ischemia 影像: {ischemia_count} 張")
    print()
    
    if bleeding_count == 0 or ischemia_count == 0:
        print("❌ 錯誤: 資料夾中沒有找到PNG影像文件")
        return False
    
    return True

def main_menu():
    """主選單"""
    if not quick_start():
        return
    
    print("請選擇操作:")
    print("1. 開始訓練CGAN並生成影像")
    print("2. 測試已存在的模型")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n請輸入選項 (1-3): ").strip()
            
            if choice == '1':
                print("\n開始訓練CGAN模型...")
                print("注意: 訓練過程可能需要較長時間，請耐心等待")
                main()
                break
                
            elif choice == '2':
                print("\n測試已存在的模型...")
                test_existing_model()
                break
                
            elif choice == '3':
                print("退出程式")
                break
                
            else:
                print("無效選項，請輸入 1、2 或 3")
                
        except KeyboardInterrupt:
            print("\n\n程式被用戶中斷")
            break
        except Exception as e:
            print(f"發生錯誤: {e}")
            break

if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        print(f"程式執行時發生錯誤: {e}")
        print("請檢查您的環境設定和依賴項")
