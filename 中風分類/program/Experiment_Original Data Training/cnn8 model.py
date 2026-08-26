# -*- coding: utf-8 -*-
"""
CT Scan Classification for Stroke Detection
(訓練程式 - 僅使用原始資料)
(*** Gemini 修改版：加入 Specificity, NPV, ROC-AUC ***)
(*** 修正版：修復 history_finetune NameError 及其他小問題 ***)
(*** 新增：95% 信賴區間 Wald + Wilson ***)
"""

# === Main Libraries ===
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import cv2
from tqdm import tqdm
import shutil

import tensorflow as tf
print(tf.__version__)  # 確認 TensorFlow 版本
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models

# ---
# ✅ (1/3) 加入計算 ROC/AUC 所需的函式庫
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
# ---

# === Configuration ===
data_dir = r"C:\Users\User\OneDrive\Desktop\Special Topic\archive\Brain_Stroke_CT_Dataset"
# cgan_dir is removed as we are not using augmented data
output_dir = r"C:\Users\User\OneDrive\Desktop\Special Topic\result\cnn8_result"

# ---
# ✅ (2/3) 定義新指標圖片的儲存資料夾
plot_save_dir = r"C:\Users\User\OneDrive\Desktop\Special Topic\result\cnn8_result"
# ---

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
print(f"Output directory created at: {output_dir}")

# ---
# 建立新指標的資料夾
os.makedirs(plot_save_dir, exist_ok=True)
print(f"Additional plots directory created at: {plot_save_dir}")
# ---

# === Get Paths ===
ischemia_path = os.path.join(data_dir, 'Ischemia', 'PNG')
hemorrhagic_path = os.path.join(data_dir, 'Hemorrhagic', 'PNG')
normal_path = os.path.join(data_dir, 'Normal', 'PNG')

normal_images = os.listdir(normal_path)
ischemia_images = os.listdir(ischemia_path)
hemorrhagic_images = os.listdir(hemorrhagic_path)

# === 原始資料分布統計 ===
print("\n--- Original Dataset Distribution ---")
original_counts = {
    "Normal": len(normal_images),
    "Ischemia": len(ischemia_images),
    "Hemorrhagic": len(hemorrhagic_images)
}
for class_name, count in original_counts.items():
    print(f"{class_name} Images Count: {count}")
print(f"Total Original Images: {sum(original_counts.values())}")
print("------------------------------------")

# === Load Images ===
imgs = []
label = []

path_lookup = {
    "Normal": normal_path,
    "Ischemia": ischemia_path,
    "Hemorrhagic": hemorrhagic_path,
}

for img_list, label_name in [(normal_images, "Normal"), (ischemia_images, "Ischemia"), (hemorrhagic_images, "Hemorrhagic")]:
    path = path_lookup[label_name]
    for img in img_list:
        img_path = os.path.join(path, img)
        imgs.append(img_path)
        label.append(label_name)

df = pd.DataFrame({"Image_path": imgs, "Label": label})
print("\nOriginal Data Loaded:")
print(df.head())

# === Plot and Save Sample Classes ===
def plot_samples(df, class_name, save_name):
    plt.figure(figsize=(12, 6))
    for i, img_path in enumerate(tqdm(df[df['Label'] == class_name]['Image_path'].head(8))):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(class_name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_name))
    plt.close()

plot_samples(df, "Normal", "sample_images_normal.png")
plot_samples(df, "Ischemia", "sample_images_ischemia.png")
plot_samples(df, "Hemorrhagic", "sample_images_hemorrhagic.png")

# 類別分佈圖
sns.countplot(data=df, x='Label', hue='Label', palette='Set2', legend=False)
plt.title("Distribution of Classes")
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

# === Split the data ===
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Label'])
print(f"\nOriginal Split: Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

# === 數據分布統計 ===
print("\n--- Data Distribution ---")
train_counts = train_df['Label'].value_counts()
val_counts = val_df['Label'].value_counts()
test_counts = test_df['Label'].value_counts()

summary_df = pd.DataFrame({
    'Training': train_counts,
    'Validation': val_counts,
    'Testing': test_counts
}).fillna(0).astype(int)
summary_df['Total'] = summary_df.sum(axis=1)
summary_df.loc['Total'] = summary_df.sum()
print(summary_df)

# === Generator ===
# 註：CNN8 是從頭訓練的簡單 CNN，不是 EfficientNet 遷移學習，
# 所以這裡改用單純的 rescale 正規化，而不是 EfficientNet 專用的 preprocess_input。
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='Image_path', y_col='Label',
    target_size=(300, 300), batch_size=32, class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='Image_path', y_col='Label',
    target_size=(300, 300), batch_size=32, class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='Image_path', y_col='Label',
    target_size=(300, 300), batch_size=32, class_mode='categorical', shuffle=False
)

# === 1. 建立 CNN8 模型 ===
inputs = layers.Input(shape=(300, 300, 3))

# 第 1 層: 卷積層 (32 filters)
x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D()(x)

# 第 2 層: 卷積層 (64 filters)
x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D()(x)

# 第 3 層: 卷積層 (128 filters)
x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D()(x)

# 第 4 層: 卷積層 (128 filters)
x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D()(x)

# 第 5 層: 卷積層 (256 filters) + GAP
x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.GlobalAveragePooling2D()(x)  # 取代 Flatten

# 第 6 層: 全連接層
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# 第 7 層: 全連接層
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# 第 8 層: 輸出層 (假設有 3 個類別)
outputs = layers.Dense(3, activation='softmax')(x)

# 命名為 CNN8_Modern
model = models.Model(inputs=inputs, outputs=outputs, name='CNN8_Modern')

# === 2. 編譯模型 ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# === 3. 設定 Callbacks ===
# 將儲存檔名改為 cnn8，避免覆蓋舊檔案
checkpoint_filepath = os.path.join(output_dir, "best_cnn8_model.keras")
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

# === 4. 開始訓練 ===
print("\n--- Starting Training ---")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[checkpoint_callback]
)
print("\n--- Training Finished ---")
print(f"✅ Best model saved to '{checkpoint_filepath}'")

# === Plot and Save acc & loss ===
# 修正: 原本誤用了未定義的 history_finetune，這裡改回實際存訓練結果的 history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(output_dir, "training_history.png"))
plt.close()

# === Evaluate model ===
print("\nLoading best model for final evaluation...")
best_model = tf.keras.models.load_model(checkpoint_filepath)

# 準確度 (Accuracy)
# 修正: 改名為 test_loss / test_accuracy，避免和上面訓練歷史中的 val_loss 變數混淆
# （這裡評估的是「測試集」，不是驗證集）
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Evaluation on Test Set: loss is {test_loss:.4f}, accuracy is {test_accuracy*100:.2f}%")

# === Report & Confusion Matrix ===
y_pred_probs = best_model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())
n_classes = len(class_names)

print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

print(f"Confusion matrix plot saved to '{os.path.join(output_dir, 'confusion_matrix.png')}'")


# ---
# ✅ (3/3) 新增指標計算 (Specificity, NPV, ROC-AUC)
# ---

# 用於儲存要寫入檔案的額外指標
report_lines_to_add = ["\n\n--- Additional Metrics (One-vs-Rest) ---"]

# === 1. 計算特異度 (Specificity) 和 陰性預測值 (NPV) ===
print("\n--- Calculating Specificity and NPV (per-class) ---")
specificities = []
npvs = []

for i in range(n_classes):
    # TP = cm[i, i] (True Positive)
    # FP (False Positive) = Sum of column i, minus TP
    FP = cm[:, i].sum() - cm[i, i]
    # FN (False Negative) = Sum of row i, minus TP
    FN = cm[i, :].sum() - cm[i, i]
    # TN (True Negative) = Total sum, minus (TP + FP + FN)
    TN = cm.sum() - (cm[i, i] + FP + FN)

    # Specificity = TN / (TN + FP)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    specificities.append(specificity)

    # NPV = TN / (TN + FN)
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    npvs.append(npv)

    print(f"Class: {class_names[i]}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  NPV (Negative Predictive Value): {npv:.4f}")

    # 準備寫入報告
    report_lines_to_add.append(f"Class: {class_names[i]}")
    report_lines_to_add.append(f"  Specificity: {specificity:.4f}")
    report_lines_to_add.append(f"  NPV:           {npv:.4f}")


# === 2. 計算並繪製 ROC-AUC 曲線 ===
print("\n--- Calculating and Plotting ROC-AUC Curve ---")

# 將真實標籤進行 One-Hot 編碼 (Binarize)
y_true_bin = label_binarize(y_true, classes=range(n_classes))

# 計算每個類別的 ROC 曲線和 AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 計算 micro-average (微觀平均) ROC 曲線和 AUC
# (將所有類別的 y_true 和 y_pred_probs 攤平)
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 繪製所有 ROC 曲線
plt.figure(figsize=(10, 8))

# 繪製 Micro-average ROC
plt.plot(fpr["micro"], tpr["micro"],
         label=f'Micro-average ROC (area = {roc_auc["micro"]:0.4f})',
         color='deeppink', linestyle=':', linewidth=4)

# 繪製每個類別的 ROC
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 繪製對角線
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# 儲存圖片到指定資料夾
roc_save_path = os.path.join(plot_save_dir, "roc_auc_curve.png")
plt.savefig(roc_save_path)
plt.close()
print(f"ROC-AUC curve plot saved to '{roc_save_path}'")

# 準備將 AUC 寫入報告
report_lines_to_add.append("\n--- ROC-AUC Scores ---")
for i in range(n_classes):
    report_lines_to_add.append(f"  {class_names[i]} AUC: {roc_auc[i]:0.4f}")
report_lines_to_add.append(f"  Micro-average AUC: {roc_auc['micro']:.4f}")

# === 4. 計算 95% 信賴區間 (Wald + Wilson) ===
print("\n--- Calculating 95% Confidence Interval (Wald + Wilson) ---")
n = np.sum(cm)
correct_predictions = np.trace(cm)
accuracy = correct_predictions / n
Z = 1.96

# Wald interval
se = math.sqrt((accuracy * (1 - accuracy)) / n)
ci_lower = accuracy - (Z * se)
ci_upper = accuracy + (Z * se)
ci_lower = max(0.0, ci_lower)
ci_upper = min(1.0, ci_upper)

# Wilson score interval（對小樣本/極端比例更穩健，可作為對照）
denom = 1 + (Z ** 2) / n
center = (accuracy + (Z ** 2) / (2 * n)) / denom
margin = (Z * math.sqrt((accuracy * (1 - accuracy) / n) + (Z ** 2) / (4 * n ** 2))) / denom
wilson_lower = max(0.0, center - margin)
wilson_upper = min(1.0, center + margin)

print(" Confidence Interval (95% CI)")
print("=" * 45)
print(f"▸ 測試總樣本數 (n) : {n}")
print(f"▸ 預測正確數量     : {correct_predictions}")
print(f"▸ 模型準確率 (Acc) : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"▸ 95% CI (Wald)    : [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"▸ 95% CI (Wilson)  : [{wilson_lower:.4f}, {wilson_upper:.4f}]")
print("-" * 45)

report_lines_to_add.append("\n--- 95% Confidence Interval ---")
report_lines_to_add.append(f"Sample size (n): {n}")
report_lines_to_add.append(f"Correct predictions: {correct_predictions}")
report_lines_to_add.append(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
report_lines_to_add.append(f"95% CI (Wald): [{ci_lower:.4f}, {ci_upper:.4f}]")
report_lines_to_add.append(f"95% CI (Wilson): [{wilson_lower:.4f}, {wilson_upper:.4f}]")


# === 5. 儲存包含所有指標的報告 ===
report_filepath = os.path.join(output_dir, "classification_report.txt")
with open(report_filepath, 'w') as f:
    # 寫入準確度 (Accuracy) 和 Loss
    f.write(f"Test Set Accuracy: {test_accuracy*100:.2f}%\n")
    f.write(f"Test Set Loss: {test_loss:.4f}\n\n")

    # 寫入標準 Classification Report
    f.write("Classification Report:\n")
    f.write(report)

    # 寫入新增的指標 (Specificity, NPV, AUC, 95% CI)
    for line in report_lines_to_add:
        f.write(line + "\n")

print(f"Classification report (with new metrics) saved to '{report_filepath}'")
print("\n--- All tasks completed and results saved. ---")