# -*- coding: utf-8 -*-
"""
模型預測與 Grad-CAM++ 可視化程式 (搭配展開式模型)
"""

# === 1. 導入函式庫 ===
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2
import os

# === 2. 設定參數 ===
# --- 請確保路徑指向新訓練出的模型 ---
model_path = r"C:\Users\USER\Desktop\CT_PRO\stroke_model_output\stroke_model_output\effnet_stroke_best.h5"
# 要測試的單張圖片路徑
img_path = r"C:\Users\USER\Desktop\strokedata\Brain_Stroke_CT_Dataset\External_Test\PNG\13477.png"
# 輸出圖片儲存的路徑
output_dir = r"C:\Users\USER\Desktop\CT_PRO\stroke_model_output\stroke_model_output\result"
# --------------------

# 模型的輸入圖片尺寸
IMG_SIZE = (224, 224)
# 類別名稱
CLASS_NAMES = ['Bleeding', 'Ischemia', 'Normal']

os.makedirs(output_dir, exist_ok=True)
print(f"輸出圖片將儲存至: {output_dir}")

# === 3. Grad-CAM++ 核心函式 (修正版) ===
def get_grad_cam_plus_plus(model, img_array, last_conv_layer_name):
    """
    計算並產生 Grad-CAM++ 熱圖 (適用於展開式模型)
    使用巢狀 GradientTape 修正高階梯度計算問題
    """
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 將 numpy 陣列轉換為 TensorFlow 張量
    img_tensor = tf.convert_to_tensor(img_array)

    # 使用巢狀 GradientTape 來計算高階梯度
    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                # 前向傳播必須在所有 tape 的上下文中執行
                conv_outputs, predictions = grad_model(img_tensor)
                
                # 確保 conv_outputs 的資料類型為 float32 以進行梯度計算
                conv_outputs = tf.cast(conv_outputs, tf.float32)

                pred_index = tf.argmax(predictions[0])
                # 對於 batch size=1 的情況，簡化 loss 的取法
                loss = predictions[0, pred_index]

            # 計算一階梯度
            first_grads = tape1.gradient(loss, conv_outputs)
            # 安全性檢查，如果無法計算梯度則拋出錯誤
            if first_grads is None:
                raise ValueError("無法計算一階梯度。請檢查模型架構與損失函數。")
        
        # 計算二階梯度
        second_grads = tape2.gradient(first_grads, conv_outputs)
        if second_grads is None:
            raise ValueError("無法計算二階梯度。模型中的某些操作可能不支援高階微分。")
            
    # 計算三階梯度
    third_grads = tape3.gradient(second_grads, conv_outputs)
    if third_grads is None:
        raise ValueError("無法計算三階梯度。")

    # 去除 batch 維度以便計算
    conv_outputs = conv_outputs[0]
    first_grads = first_grads[0]
    second_grads = second_grads[0]
    third_grads = third_grads[0]

    # Grad-CAM++ 權重計算
    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1))
    
    alpha_denominator = 2 * second_grads + third_grads * global_sum
    # 避免除以零
    alpha_denominator = tf.where(alpha_denominator == 0.0, 1e-7, alpha_denominator)
    
    alphas = second_grads / alpha_denominator
    weights = tf.reduce_sum(alphas * tf.maximum(first_grads, 0), axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

    # 熱圖後處理
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = (heatmap * 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap

# === 4. 主程式執行流程 ===
try:
    print(f"正在載入模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"正在讀取圖片: {img_path}")
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    preprocessed_img = preprocess_input(np.expand_dims(img_array, axis=0))
    
    print("正在進行預測...")
    predictions = model.predict(preprocessed_img, verbose=0)
    pred_index = np.argmax(predictions[0])
    pred_class = CLASS_NAMES[pred_index]
    pred_confidence = predictions[0][pred_index]
    print(f"預測結果: {pred_class} (信心度: {pred_confidence:.2%})")
    
    print("正在產生 Grad-CAM++ 熱圖...")
    
    # 在展開的模型中，可以直接反向尋找最外層的 Conv2D 層
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            last_conv_layer_name = layer.name
            break
            
    if not last_conv_layer_name:
        raise ValueError("模型中未找到 Conv2D 層。")

    print(f"目標卷積層: {last_conv_layer_name}")
    heatmap = get_grad_cam_plus_plus(model, preprocessed_img, last_conv_layer_name)
    
    original_img_for_display = cv2.resize(cv2.imread(img_path), IMG_SIZE)
    superimposed_img = cv2.addWeighted(original_img_for_display, 0.6, heatmap, 0.4, 0)
    
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    output_filename = f"{base_filename}_gradcampp_{pred_class}.png"
    output_filepath = os.path.join(output_dir, output_filename)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"prediction: {pred_class} ({pred_confidence:.2%})", fontsize=16)
    ax1.imshow(cv2.cvtColor(original_img_for_display, cv2.COLOR_BGR2RGB))
    ax1.set_title("original image")
    ax1.axis('off')
    ax2.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Grad-CAM++ visualization")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(output_filepath)
    print(f"✅ Grad-CAM++ 圖片已成功儲存至: {output_filepath}")
    plt.show()

except Exception as e:
    print(f"發生未預期的錯誤: {e}")