# -*- coding: utf-8 -*-
"""
CT Scan Classification - Knowledge Distillation
Teacher: EfficientNetB3 → Student: ResNet20
任務: 3分類 (Normal / Ischemia / Hemorrhagic)
"""

# ============================================================
# 1. 環境設定
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from itertools import cycle

print("✓ 環境設定完成！")
print(f"TensorFlow 版本: {tf.__version__}")

# ============================================================
# 2. 設定隨機種子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

device_name = tf.test.gpu_device_name()
print(f"使用裝置: {device_name if device_name else '/CPU:0'}")

# ============================================================
# 3. 路徑設定
# ============================================================
DATA_DIR     = r'archive\Brain_Stroke_CT_Dataset'
TEACHER_PATH = r"C:\Users\User\OneDrive\Desktop\Special Topic\result\b3_result\best_finetuned_model.keras"
OUTPUT_DIR   = r"b3_distillation_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ 輸出資料夾: {OUTPUT_DIR}")

# ============================================================
# 4. 資料載入 (data.py 對應區塊)
# ============================================================
def get_ct_dataloaders(data_dir, batch_size=32):
    ischemia_path    = os.path.join(data_dir, 'Ischemia',    'PNG')
    hemorrhagic_path = os.path.join(data_dir, 'Hemorrhagic', 'PNG')
    normal_path      = os.path.join(data_dir, 'Normal',      'PNG')

    imgs, labels = [], []
    for label_name in ["Normal", "Ischemia", "Hemorrhagic"]:
        path = locals()[f"{label_name.lower()}_path"]
        for img in os.listdir(path):
            imgs.append(os.path.join(path, img))
            labels.append(label_name)

    df = pd.DataFrame({"Image_path": imgs, "Label": labels})

    # 切分：train 64% / val 16% / test 20%
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Label'])
    train_df, val_df  = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['Label'])

    print(f"\n資料分布：Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_loader = datagen.flow_from_dataframe(
        train_df, x_col='Image_path', y_col='Label',
        target_size=(300, 300), batch_size=batch_size,
        class_mode='categorical', shuffle=True)

    val_loader = datagen.flow_from_dataframe(
        val_df, x_col='Image_path', y_col='Label',
        target_size=(300, 300), batch_size=batch_size,
        class_mode='categorical', shuffle=False)

    test_loader = datagen.flow_from_dataframe(
        test_df, x_col='Image_path', y_col='Label',
        target_size=(300, 300), batch_size=batch_size,
        class_mode='categorical', shuffle=False)

    return train_loader, val_loader, test_loader, len(train_df), 3

train_loader, val_loader, test_loader, train_size, num_classes = get_ct_dataloaders(
    data_dir=DATA_DIR, batch_size=32)

print(f"類別數: {num_classes}")
print(f"訓練樣本: {train_size}")
CLASS_NAMES = list(train_loader.class_indices.keys())
print(f"類別名稱: {CLASS_NAMES}")

# ============================================================
# 5. 模型定義 (models.py 對應區塊)
# ============================================================

# --- Teacher Model ---
def get_teacher_model(model_path):
    model = tf.keras.models.load_model(model_path)
    model.trainable = False
    print(f"✓ Teacher 模型載入完成: {model_path}")
    return model

# --- Student Model: ResNet20 ---
def resnet_block(x, filters, kernel_size=3, stride=1, use_shortcut=False):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=1,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    if use_shortcut or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def get_student_model(input_shape=(300, 300, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # Initial Conv
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Stage 1: 16 filters
    x = resnet_block(x, 16, use_shortcut=True)
    x = resnet_block(x, 16)
    x = resnet_block(x, 16)

    # Stage 2: 32 filters
    x = resnet_block(x, 32, stride=2, use_shortcut=True)
    x = resnet_block(x, 32)
    x = resnet_block(x, 32)

    # Stage 3: 64 filters
    x = resnet_block(x, 64, stride=2, use_shortcut=True)
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    # Output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)

    return models.Model(inputs, outputs, name='ResNet20_Student')

teacher_model = get_teacher_model(TEACHER_PATH)
student_model = get_student_model(num_classes=num_classes)

# ============================================================
# 6. 模型比較 (utils.py 對應區塊)
# ============================================================
def compare_models(teacher, student):
    t_params = teacher.count_params()
    s_params = student.count_params()
    compression_ratio = t_params / s_params
    percentage = (s_params / t_params) * 100

    print("=" * 60)
    print("模型參數比較")
    print("=" * 60)
    print(f"教師模型 (EfficientNetB3): {t_params:,} 個參數")
    print(f"學生模型 (ResNet20):       {s_params:,} 個參數")
    print(f"壓縮比例: {compression_ratio:.2f}x")
    print(f"學生模型僅為教師模型的 {percentage:.2f}%")
    print("=" * 60)

compare_models(teacher_model, student_model)

# ============================================================
# 7. 知識蒸餾訓練 (distillation.py 對應區塊)
# ============================================================
class DistillationTrainer(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, alpha, temperature):
        super().compile(
            optimizer=optimizer,
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
            run_eagerly=True
        )
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()

    def train_step(self, data):
        x, y = data

        # Teacher 預測 (不訓練)
        teacher_preds = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)

            # Hard loss: 學生 vs 真實標籤
            student_loss = self.student_loss_fn(y, student_preds)

            # Soft loss: 學生 vs 老師 (用溫度軟化)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_preds / self.temperature, axis=1),
                tf.nn.softmax(student_preds / self.temperature, axis=1)
            )

            # 總損失
            loss = (self.alpha * student_loss +
                    (1 - self.alpha) * distillation_loss * (self.temperature ** 2))

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, tf.nn.softmax(student_preds, axis=1))

        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results

    def test_step(self, data):
        x, y = data
        student_preds = self.student(x, training=False)
        student_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y, student_preds)
        self.compiled_metrics.update_state(y, tf.nn.softmax(student_preds, axis=1))
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = student_loss
        return results

    def get_config(self):
        return {}

# ============================================================
# 8. 蒸餾設定與訓練
# ============================================================
TEMPERATURE = 3.0
ALPHA       = 0.7
NUM_EPOCHS  = 50  # 建議 50+，示範可改 10

trainer = DistillationTrainer(student=student_model, teacher=teacher_model)
trainer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    alpha=ALPHA,
    temperature=TEMPERATURE
)

print(f"\n知識蒸餾配置：")
print(f"  溫度參數 (Temperature): {TEMPERATURE}")
print(f"  Alpha (Hard loss 權重): {ALPHA}")
print(f"  訓練輪數: {NUM_EPOCHS}")

# Checkpoint 儲存最佳模型 (只存 student 子模型,避免存到整個 DistillationTrainer)
class SaveBestStudentCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == 'min' else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (current < self.best) if self.mode == 'min' else (current > self.best)
        if improved:
            self.best = current
            self.model.student.save(self.filepath)
            print(f"\nEpoch {epoch+1}: {self.monitor} improved to {current:.4f}, saving student model to {self.filepath}")

checkpoint_cb = SaveBestStudentCallback(
    filepath=os.path.join(OUTPUT_DIR, 'best_student_model.keras'),
    monitor='val_loss',
    mode='min'
)

print("\n--- 開始知識蒸餾訓練 ---")
history = trainer.fit(
    train_loader,
    validation_data=val_loader,
    epochs=NUM_EPOCHS,
    callbacks=[checkpoint_cb],
    verbose=1
)
print("\n--- 訓練完成 ---")

# 儲存最終學生模型
student_model.save(os.path.join(OUTPUT_DIR, 'student_model_distilled.keras'))
print("✓ 學生模型已儲存！")

# ============================================================
# 9. 繪製訓練曲線 (utils.py - plot_training_curves)
# ============================================================
def plot_training_curves(history, save_dir):
    acc  = history.history['accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    print(f"✓ 訓練曲線已儲存")

plot_training_curves(history, OUTPUT_DIR)

# ============================================================
# 10. 完整評估 (含 Specificity, NPV, ROC-AUC)
# ============================================================
print("\n--- 載入最佳學生模型進行評估 ---")
best_student_path = os.path.join(OUTPUT_DIR, 'best_student_model.keras')
best_student = tf.keras.models.load_model(best_student_path)
print(f"✓ 已載入最佳學生模型: {best_student_path}")

# 取得預測結果
y_pred_logits = best_student.predict(test_loader)
y_pred_probs  = tf.nn.softmax(y_pred_logits, axis=1).numpy()
y_pred        = np.argmax(y_pred_probs, axis=1)
y_true        = test_loader.classes
n_classes     = len(CLASS_NAMES)

# --- Accuracy ---
# 用 Teacher 評估
teacher_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
_, teacher_acc = teacher_model.evaluate(test_loader, verbose=0)

# 用 Student 評估 (logits)
best_student.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
_, student_acc = best_student.evaluate(test_loader, verbose=0)

print("=" * 60)
print("最終效能比較")
print("=" * 60)
print(f"教師模型 (EfficientNetB3) 準確率: {teacher_acc*100:.2f}%")
print(f"學生模型 (ResNet20) 準確率:       {student_acc*100:.2f}%")
print(f"效能保留率: {100 * student_acc / teacher_acc:.2f}%")
print("=" * 60)

# --- Classification Report ---
print("\nClassification Report (Student Model):")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
print(report)

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Student Model - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()
print("✓ Confusion Matrix 已儲存")

# --- Specificity & NPV ---
print("\n--- Specificity & NPV (per-class) ---")
specificities, npvs = [], []
report_lines = ["\n--- Additional Metrics (One-vs-Rest) ---"]

for i in range(n_classes):
    FP = cm[:, i].sum() - cm[i, i]
    FN = cm[i, :].sum() - cm[i, i]
    TN = cm.sum() - (cm[i, i] + FP + FN)

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    npv         = TN / (TN + FN) if (TN + FN) > 0 else 0
    specificities.append(specificity)
    npvs.append(npv)

    print(f"Class: {CLASS_NAMES[i]}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  NPV:         {npv:.4f}")
    report_lines.append(f"Class: {CLASS_NAMES[i]}")
    report_lines.append(f"  Specificity: {specificity:.4f}")
    report_lines.append(f"  NPV:         {npv:.4f}")

# --- ROC-AUC ---
print("\n--- ROC-AUC Curve ---")
y_true_bin = label_binarize(y_true, classes=range(n_classes))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(10, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.4f})',
         color='deeppink', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC - {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Student Model (ResNet20) - ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_auc_curve.png'))
plt.close()
print("✓ ROC-AUC 曲線已儲存")

report_lines.append("\n--- ROC-AUC Scores ---")
for i in range(n_classes):
    report_lines.append(f"  {CLASS_NAMES[i]} AUC: {roc_auc[i]:.4f}")
report_lines.append(f"  Micro-average AUC: {roc_auc['micro']:.4f}")

# ============================================================
# 11. 儲存完整報告
# ============================================================
report_path = os.path.join(OUTPUT_DIR, 'distillation_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("Knowledge Distillation Report\n")
    f.write("Teacher: EfficientNetB3 → Student: ResNet20\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Teacher Accuracy: {teacher_acc*100:.2f}%\n")
    f.write(f"Student Accuracy: {student_acc*100:.2f}%\n")
    f.write(f"Performance Retention: {100 * student_acc / teacher_acc:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    for line in report_lines:
        f.write(line + "\n")

print(f"✓ 完整報告已儲存至: {report_path}")
print("\n=== 所有任務完成！===")