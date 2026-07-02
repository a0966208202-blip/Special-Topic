class Constants:
    # --- 新增您的資料集名稱 ---
    STROKE_CT = "stroke_ct"

    # --- cGAN 相關常數 ---
    CGAN_N_CLASSES = 2  # *** 修改為 2，因為只擴增 Bleeding 和 Ischemia ***
    CGAN_Z_DIM = 64
    # TTUR：為 G/D 設定不同學習率
    C_GAN_LR_G = 2e-4
    C_GAN_LR_D = 5e-5
    C_GAN_BATCH_SIZE = 64 # 您可以根據您的 GPU VRAM 調整此數值，例如 64 或 32
    CGAN_EPOCH = 500 # 訓練週期，可以先設為 100-200 來測試

    # 每個 batch 中，G 的更新次數（D 固定 1 次）
    CGAN_G_UPDATES_PER_D = 2

    # 標籤平滑
    LABEL_SMOOTH_TRUE = 0.9
    LABEL_SMOOTH_FAKE = 0.1

    # --- 檔案路徑 (可以保留，但目前 cGAN 流程不會直接使用) ---
    C_GAN_TRAIN_IMAGE_PATH = "./C_GAN_Images/Training_Images_before_Training_{}"
    C_GAN_TRAIN_IMAGE_PATH_AFTER_TRAINING = "./C_GAN_Images/Training_Images_after_Training_%d_{}"
