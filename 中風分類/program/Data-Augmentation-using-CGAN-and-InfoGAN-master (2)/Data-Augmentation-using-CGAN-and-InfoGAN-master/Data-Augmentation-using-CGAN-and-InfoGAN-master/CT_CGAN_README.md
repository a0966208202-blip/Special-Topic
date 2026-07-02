# CT影像CGAN資料擴增指南

本專案使用Conditional GAN (CGAN) 來擴增Bleeding和Ischemia CT影像，以改善醫學影像分類模型的性能。

## 📁 資料夾結構

請確保您的資料夾結構如下：

```
Data-Augmentation-using-CGAN-and-InfoGAN-master/
├── train/
│   ├── Bleeding/          # 出血影像 (PNG格式)
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── Ischemia/          # 缺血影像 (PNG格式)
│       ├── image1.png
│       ├── image2.png
│       └── ...
├── test/                  # 測試影像 (可選)
│   ├── Bleeding/
│   └── Ischemia/
└── ... (其他程式文件)
```

## 🚀 快速開始

### 方法1: 使用簡化腳本 (推薦)

```bash
python run_ct_augmentation.py
```

此腳本會自動檢查您的資料夾結構，並提供互動式選單。

### 方法2: 直接執行訓練腳本

```bash
python train_cgan_ct.py
```

## ⚙️ 程式參數設定

在 `CONSTANTS.py` 中可以調整以下參數：

```python
# CGAN相關參數
CGAN_N_CLASSES = 2          # 類別數量 (Bleeding, Ischemia)
CGAN_Z_DIM = 64             # 雜訊維度
C_GAN_LR = 0.0002           # 學習率
C_GAN_BATCH_SIZE = 128      # 批次大小 (可根據GPU記憶體調整)
CGAN_EPOCH = 500            # 訓練週期
```

## 📊 訓練過程

1. **資料載入**: 自動載入train資料夾中的影像
2. **模型訓練**: 訓練CGAN生成器和判別器
3. **影像生成**: 為每個類別生成1000張新影像
4. **資料擴增**: 將生成的影像與原始資料合併

## 📈 輸出結果

訓練完成後，您會得到以下文件：

- `checkpoint/C_GAN/C_GAN_FINAL_MODEL_stroke_ct.pt`: 訓練好的模型
- `C_GAN_generate_datasets/2k_image_set_stroke_ct_noise_1.pt`: 生成的影像資料
- `C_GAN_Images/Generated_CT_Images_stroke_ct.png`: 生成的影像預覽

## 🔧 系統需求

- Python 3.7+
- PyTorch 1.8+
- CUDA (可選，用於GPU加速)
- 其他依賴項請參考requirements.txt

## 📝 使用範例

### 訓練新模型
```python
from train_cgan_ct import main
main()
```

### 測試現有模型
```python
from train_cgan_ct import test_existing_model
test_existing_model()
```

### 使用擴增後的資料
```python
from build_dataset import DataLoader
from augment_dataset import AugmentedDataloader

# 載入原始資料
dataloader = DataLoader("stroke_ct")

# 建立擴增資料載入器
augmented_dataloader = AugmentedDataloader()
aug_train, aug_test = augmented_dataloader.get_augmented_dataloader(
    original_dataloader=dataloader,
    dataset_name="stroke_ct"
)
```

## ⚠️ 注意事項

1. **GPU記憶體**: 如果遇到GPU記憶體不足，請降低 `C_GAN_BATCH_SIZE`
2. **訓練時間**: 完整訓練可能需要數小時，建議先用較少的epoch測試
3. **影像品質**: 生成的影像品質取決於原始資料的品質和數量
4. **資料平衡**: 確保Bleeding和Ischemia影像數量大致平衡

## 🐛 常見問題

### Q: 出現 "CUDA out of memory" 錯誤
A: 降低批次大小，將 `C_GAN_BATCH_SIZE` 改為 64 或 32

### Q: 生成的影像品質不佳
A: 增加訓練週期或調整學習率

### Q: 找不到資料夾
A: 確保資料夾結構正確，且PNG檔案位於正確位置

## 📞 支援

如果遇到問題，請檢查：
1. 資料夾結構是否正確
2. 影像檔案是否為PNG格式
3. Python環境和依賴項是否正確安裝

---

**祝您使用愉快！** 🎉
