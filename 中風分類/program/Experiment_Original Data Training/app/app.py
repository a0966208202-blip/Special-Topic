# -*- coding: utf-8 -*-
"""
CT Scan Classification for Stroke Detection
Gradio Application with Grad-CAM++ Visualization (Simplified - No Cropping)
"""

# === Main Libraries ===
import numpy as np
import os
import cv2
import zipfile
import gradio as gr
from PIL import Image

import tensorflow as tf
print(tf.__version__)  # Confirm TensorFlow version
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. Global Settings & Model Loading ---
IMG_HEIGHT, IMG_WIDTH = 224, 224
# Class order must be consistent with the training ImageDataGenerator (alphabetical)
CLASS_NAMES = ['Bleeding', 'Ischemia', 'Normal']
try:
    model_path = r"C:\Users\USER\Desktop\中風分類\實驗_原始資料訓練\result_v2_Dropout_ClassWeight\best_finetuned_model.keras"
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error: Could not load model from '{model_path}'")
    print(f"Details: {e}")
    exit()

last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer_name = layer.name
        break
if last_conv_layer_name:
    print(f"Found last convolutional layer for Grad-CAM++: {last_conv_layer_name}")
else:
    print("Warning: No convolutional layer found in the model. Grad-CAM++ may not work.")

# --- 2. Core Function Definitions (Simplified) ---

def preprocess_image(pil_image):
    """
    Preprocesses an uploaded PIL image to meet model input requirements.
    """
    image = pil_image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image).astype("float32")
    image_array = preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

def get_grad_cam_plus_plus(model, img_array, layer_name):
    """
    Calculate and generate Grad-CAM++ heatmap (適用於展開式模型)
    Using nested GradientTape to fix higher-order gradient calculation issues
    """
    grad_model = Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )

    # Convert numpy array to TensorFlow tensor
    img_tensor = tf.convert_to_tensor(img_array)

    # Use nested GradientTape to calculate higher-order gradients
    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                # Forward pass must be executed in all tape contexts
                conv_outputs, predictions = grad_model(img_tensor)
                
                # Ensure conv_outputs data type is float32 for gradient calculation
                conv_outputs = tf.cast(conv_outputs, tf.float32)

                pred_index = tf.argmax(predictions[0])
                # For batch size=1 case, simplify loss calculation
                loss = predictions[0, pred_index]

            # Calculate first-order gradients
            first_grads = tape1.gradient(loss, conv_outputs)
            # Safety check, if gradient calculation fails, raise error
            if first_grads is None:
                raise ValueError("無法計算一階梯度。請檢查模型架構與損失函數。")
        
        # Calculate second-order gradients
        second_grads = tape2.gradient(first_grads, conv_outputs)
        if second_grads is None:
            raise ValueError("無法計算二階梯度。模型中的某些操作可能不支援高階微分。")
            
    # Calculate third-order gradients
    third_grads = tape3.gradient(second_grads, conv_outputs)
    if third_grads is None:
        raise ValueError("無法計算三階梯度。")

    # Remove batch dimension for calculation
    conv_outputs = conv_outputs[0]
    first_grads = first_grads[0]
    second_grads = second_grads[0]
    third_grads = third_grads[0]

    # Grad-CAM++ weight calculation
    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1))
    
    alpha_denominator = 2 * second_grads + third_grads * global_sum
    # Avoid division by zero
    alpha_denominator = tf.where(alpha_denominator == 0.0, 1e-7, alpha_denominator)
    
    alphas = second_grads / alpha_denominator
    weights = tf.reduce_sum(alphas * tf.maximum(first_grads, 0), axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

    # Heatmap post-processing
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = (heatmap * 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap

# MODIFIED: Simplified predict_and_visualize function
def predict_and_visualize_gradcampp(uploaded_image):
    if uploaded_image is None:
        return "請先上傳一張 CT 影像。", None, None

    # --- Simplified Preprocessing ---
    # The uploaded image is a numpy array (RGB), convert to PIL Image
    pil_image = Image.fromarray(uploaded_image)
    # --- End Simplified Preprocessing ---

    # 1. Preprocess the image for the model
    processed_img_array = preprocess_image(pil_image)

    # 2. Model Prediction
    predictions = model.predict(processed_img_array)
    pred_index = np.argmax(predictions[0])
    pred_class = CLASS_NAMES[pred_index]
    pred_confidence = np.max(predictions[0])
    result_text = f"預測結果: {pred_class}\n信賴度: {pred_confidence:.2%}"

    # 3. Generate Grad-CAM++
    # Resize the original uploaded image to match model input for visualization
    resized_original_img = cv2.resize(uploaded_image, (IMG_WIDTH, IMG_HEIGHT))
    # Convert to BGR for OpenCV colormap functions
    original_img_for_cam = cv2.cvtColor(resized_original_img, cv2.COLOR_RGB2BGR)
    
    heatmap = get_grad_cam_plus_plus(model, processed_img_array, last_conv_layer_name)
    superimposed_img = cv2.addWeighted(original_img_for_cam, 0.6, heatmap, 0.4, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # 4. Prepare files for download
    with open("prediction_result.txt", "w", encoding="utf-8") as f:
        f.write(result_text)
    Image.fromarray(superimposed_img_rgb).save("grad_campp_result.png")
    with zipfile.ZipFile("results.zip", "w") as zipf:
        zipf.write("prediction_result.txt")
        zipf.write("grad_campp_result.png")
    os.remove("prediction_result.txt")
    os.remove("grad_campp_result.png")

    return result_text, superimposed_img_rgb, "results.zip"

# --- 3. Gradio Interface Design (Unchanged) ---
medical_theme_css = """
body {
    background-image: url('https://i.ibb.co/6wF2g2S/medical-bg.png');
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Georgia', 'Times New Roman', serif !important;
}
.gradio-container {
    max-width: 1000px !important; margin: auto; padding-top: 1.5rem;
    background-color: rgba(255, 255, 255, 0.92); border-radius: 15px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
}
#title_md {
    text-align: center; color: #1e3a8a; font-size: 2.5em !important;
    font-weight: bold; padding-bottom: 0px;
}
#subtitle_md {
    text-align: center; color: #374151; font-size: 1.1em !important;
    margin-top: -15px;
}
.gr-button {
    background-color: #2563eb; color: white; border-radius: 8px; font-weight: bold;
}
.gr-button:hover { background-color: #1d4ed8; }
#output_box {
    border-left: 3px solid #3b82f6; background-color: rgba(249, 250, 251, 0.8);
    padding: 1rem; border-radius: 8px;
}
#output_text_label .label-text {
    color: #1e3a8a !important; font-size: 1.2em !important; font-weight: bold;
}
#warning_md {
    background-color: #fff1f2; color: #9f1239; padding: 1rem; border-radius: 8px;
    border: 1px solid #fecaca; text-align: center;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=medical_theme_css) as demo:
    gr.Markdown("# AI 腦部 CT 影像輔助判讀系統 (Grad-CAM++)", elem_id="title_md")
    gr.Markdown("上傳您的腦部 CT 掃描影像，系統將預測其類別並使用 Grad-CAM++ 標示出 AI 模型關注的區域。", elem_id="subtitle_md")
    gr.HTML("<hr style='margin: 20px 0;'>")
    with gr.Row(equal_height=False, variant='panel'):
        with gr.Column(scale=2, min_width=300):
            input_image = gr.Image(type="numpy", label="上傳 CT 影像", height=450)
            upload_button = gr.Button("開始分析", variant="primary", scale=1)
        with gr.Column(scale=3, min_width=450):
            with gr.Group(elem_id="output_box"):
                output_text = gr.Textbox(label="AI 分析結果", elem_id="output_text_label", interactive=False, lines=2)
                output_image = gr.Image(type="numpy", label="AI Visualization (Grad-CAM++)", height=350)
                download_button = gr.File(label="下載分析報告 (.zip)", visible=False, scale=1)
    gr.Markdown(
        """
        ---
        **⚠️ 重要聲明** 此系統為基於深度學習模型之學術研究專案，其預測結果**僅供學術參考**。  
        **不可**作為任何形式的專業醫療診斷、建議或治療依據。  
        所有醫療相關決策，請務必諮詢合格的專業醫師。
        """,
        elem_id="warning_md"
    )

    def wrapper_predict(image):
        if image is None:
            gr.Warning("請先上傳圖片再進行分析！")
            return None, None, gr.File(visible=False)
        # Call the simplified function
        result, cam_img, zip_path = predict_and_visualize_gradcampp(image)
        # Check for error message from predict function
        if cam_img is None:
            return result, None, gr.File(visible=False)
        return result, cam_img, gr.File(value=zip_path, visible=True)

    upload_button.click(
        fn=wrapper_predict,
        inputs=[input_image],
        outputs=[output_text, output_image, download_button]
    )

# --- 4. Launch the Application ---
if __name__ == "__main__":
    print("啟動 Gradio 介面...")
    demo.launch(share=True)