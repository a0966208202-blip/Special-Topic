# -*- coding: utf-8 -*-
"""
Batch CT Scan Classification for Stroke Detection
Automatic Prediction & Grad-CAM++ Visualization (No Gradio)
"""

import numpy as np
import os
import cv2
import csv
from PIL import Image, ImageDraw, ImageFont # 導入 ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. Global Settings ---
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_NAMES = ['Bleeding', 'Ischemia', 'Normal'] # 注意：模型訓練時應保持 Bleeding，但顯示時轉換為 Hemorrhagic

# Paths
INPUT_DIR = r"C:\Users\USER\Desktop\strokedata\Brain_Stroke_CT_Dataset\External_Test\PNG"
OUTPUT_DIR = r"C:\Users\USER\Desktop\CT_PRO\實驗_加入資料增強1015\result"
MODEL_PATH = r"C:\Users\USER\Desktop\CT_PRO\實驗_加入資料增強1015\result\best_finetuned_model.keras"

# Create output folder if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Load Model ---
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Find last Conv layer for Grad-CAM++
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer_name = layer.name
        break
if last_conv_layer_name:
    print(f"Using last conv layer: {last_conv_layer_name}")
else:
    raise ValueError("No convolutional layer found in the model.")

# --- 3. Preprocessing Function ---
def preprocess_image(pil_image):
    image = pil_image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image).astype("float32")
    image_array = preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

# --- 4. Grad-CAM++ ---
def get_grad_cam_plus_plus(model, img_array, layer_name):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    conv_outputs = conv_outputs[0]
    grads = grads[0]

    first_derivative = tf.exp(class_channel) * grads
    second_derivative = tf.exp(class_channel) * grads * grads
    third_derivative = tf.exp(class_channel) * grads * grads * grads

    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1))
    alpha_num = second_derivative
    alpha_denom = (2.0 * second_derivative) + (global_sum * third_derivative)
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, 1e-7)
    alphas = alpha_num / alpha_denom
    alpha_normalization_constant = tf.reduce_sum(alphas, axis=(0, 1))
    alphas /= alpha_normalization_constant
    weights = tf.maximum(first_derivative, 0.0)
    deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=(0, 1))
    grad_cam_map = tf.reduce_sum(deep_linearization_weights * conv_outputs, axis=2)

    heatmap = tf.maximum(grad_cam_map, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap /= max_val
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = (heatmap * 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

# --- 5. Prediction Loop ---
results_csv_path = os.path.join(OUTPUT_DIR, "prediction_result.csv")
with open(results_csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "Prediction", "Confidence"])

    file_list = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")]
    print(f"🔍 Found {len(file_list)} images.")

    for filename in file_list:
        file_path = os.path.join(INPUT_DIR, filename)
        pil_image = Image.open(file_path)

        # Preprocess
        img_array = preprocess_image(pil_image)

        # Prediction
        predictions = model.predict(img_array, verbose=0)
        pred_index = np.argmax(predictions[0])
        pred_class_raw = CLASS_NAMES[pred_index] # 儲存原始預測類別
        pred_confidence = float(np.max(predictions[0]))

        # Display text label mapping (Bleeding -> Hemorrhagic)
        display_class = "Hemorrhagic" if pred_class_raw == "Bleeding" else pred_class_raw

        # Grad-CAM++
        resized_original_img = cv2.resize(np.array(pil_image), (IMG_WIDTH, IMG_HEIGHT))
        original_img_for_cam = cv2.cvtColor(resized_original_img, cv2.COLOR_RGB2BGR)
        heatmap = get_grad_cam_plus_plus(model, img_array, last_conv_layer_name)
        superimposed_img = cv2.addWeighted(original_img_for_cam, 0.6, heatmap, 0.4, 0)
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB) # 轉回 RGB 以便 PIL 處理

        # --- 在圖片上添加文字 ---
        # 轉換回 PIL Image 方便繪圖
        pil_superimposed_img = Image.fromarray(superimposed_img_rgb)
        draw = ImageDraw.Draw(pil_superimposed_img)

        # 設定字體 (如果沒有 'arial.ttf'，可以換成其他系統字體，或下載一個 .ttf 文件)
        try:
            # 嘗試載入更常見的字體或指定路徑
            font = ImageFont.truetype("arial.ttf", 16) # 大小可以調整
        except IOError:
            print("Warning: 'arial.ttf' not found, using default font.")
            font = ImageFont.load_default() # 使用預設字體

        text_pred = f"Pred: {display_class}"
        text_conf = f"Conf: {pred_confidence:.2%}" # 將信心度也顯示出來

        # 文字顏色 (白色)
        text_color = (255, 255, 255)

        # 文字位置 (調整至圖片上方邊緣)
        # 您可能需要根據圖片內容和文字長度微調這些位置
        draw.text((10, 5), text_pred, font=font, fill=text_color)
        draw.text((10, 25), text_conf, font=font, fill=text_color)


        # Save Grad-CAM image
        output_image_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_GradCAM.png")
        pil_superimposed_img.save(output_image_path) # 儲存 PIL Image

        # Write CSV row
        writer.writerow([filename, display_class, f"{pred_confidence:.4f}"])

        print(f"✅ Processed: {filename} -> {display_class} ({pred_confidence:.2%})")

print(f"\n📄 All results saved to: {results_csv_path}")
print(f"🖼️ Grad-CAM images saved to: {OUTPUT_DIR}")