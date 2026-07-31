# -*- coding: utf-8 -*-
"""
CT Scan Classification - Multi-Source Knowledge Distillation
Teacher1 (EfficientNetB3) + Teacher2 (EfficientNetB0) + TA (ResNet20)
    → 同時指導 →  Student (CNN8)
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
DATA_DIR      = r'archive\Brain_Stroke_CT_Dataset'
TEACHER1_PATH = r"C:\Users\User\OneDrive\Desktop\Special Topic\b3_result\best_finetuned_model.keras"
TEACHER2_PATH = r"C:\Users\User\OneDrive\Desktop\Special Topic\result\best_finetuned_model.keras"
TA_PATH       = r"C:\Users\User\OneDrive\Desktop\Special Topic\ta_distillation_result\best_ta_model.keras"
OUTPUT_DIR    = r"teacher ta_distillation_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ 輸出資料夾: {OUTPUT_DIR}")

# 資料 pipeline 統一用這個尺寸 (flow_from_dataframe 的 target_size)
PIPELINE_SIZE = 300

# 各模型「原本訓練時」用的輸入尺寸；跟 PIPELINE_SIZE 不同的話，
# get_pretrained_model() 會自動在模型前面加一層 Resizing 做轉換。
# EfficientNetB0 預設吃 224x224，EfficientNetB3 / TA(ResNet20) 這裡是用 300x300 訓練的。
TEACHER1_INPUT_SIZE = 300
TEACHER2_INPUT_SIZE = 224   # <-- 找到問題的地方：B0 是 224x224，不是 300x300
TA_INPUT_SIZE        = 300

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

# --- Teacher / TA Model (都是已經訓練好、直接載入、凍結) ---
def get_pretrained_model(model_path, name="model",
                          native_size=None, pipeline_size=PIPELINE_SIZE):
    """
    載入已訓練好的模型並凍結。
    如果該模型原本訓練時的輸入尺寸 (native_size) 跟資料 pipeline 的尺寸
    (pipeline_size) 不同，會自動包一層 Resizing，讓外部呼叫時永遠可以
    直接餵 pipeline_size 的圖片，內部自動轉成該模型需要的尺寸。
    """
    model = tf.keras.models.load_model(model_path)
    model.trainable = False

    if native_size is not None and native_size != pipeline_size:
        wrapped_input = layers.Input(shape=(pipeline_size, pipeline_size, 3))
        resized = layers.Resizing(native_size, native_size)(wrapped_input)
        wrapped_output = model(resized, training=False)
        model = models.Model(wrapped_input, wrapped_output,
                              name=f"{model.name}_resized_wrapper")
        model.trainable = False
        print(f"✓ {name} 載入完成 (輸入自動從 {pipeline_size}x{pipeline_size} "
              f"resize 成 {native_size}x{native_size}): {model_path}")
    else:
        print(f"✓ {name} 載入完成: {model_path}")

    return model

# --- Student Model: CNN8 (8層卷積的簡易CNN) ---
def get_student_model(input_shape=(300, 300, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    x = inputs
    filters_seq = [16, 16, 32, 32, 64, 64, 128, 128]  # 共 8 層 Conv
    for i, f in enumerate(filters_seq):
        x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if i % 2 == 1:  # 每兩層 conv 做一次下採樣
            x = layers.MaxPooling2D(2, padding='same')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)   # 輸出 logits (不加 softmax)，配合 from_logits=True

    return models.Model(inputs, outputs, name='CNN8_Student')

teacher1_model = get_pretrained_model(TEACHER1_PATH, name="Teacher1 (EfficientNetB3)",
                                       native_size=TEACHER1_INPUT_SIZE)
teacher2_model = get_pretrained_model(TEACHER2_PATH, name="Teacher2 (EfficientNetB0)",
                                       native_size=TEACHER2_INPUT_SIZE)
ta_model       = get_pretrained_model(TA_PATH,       name="TA (ResNet20)",
                                       native_size=TA_INPUT_SIZE)
student_model  = get_student_model(num_classes=num_classes,
                                    input_shape=(PIPELINE_SIZE, PIPELINE_SIZE, 3))

# ============================================================
# 6. 模型比較 (utils.py 對應區塊)
# ============================================================
def compare_models(teacher1, teacher2, ta, student):
    t1_params = teacher1.count_params()
    t2_params = teacher2.count_params()
    ta_params = ta.count_params()
    s_params  = student.count_params()

    print("=" * 60)
    print("模型參數比較")
    print("=" * 60)
    print(f"Teacher1 (EfficientNetB3): {t1_params:,} 個參數")
    print(f"Teacher2 (EfficientNetB0): {t2_params:,} 個參數")
    print(f"TA (ResNet20):             {ta_params:,} 個參數")
    print(f"Student (CNN8):            {s_params:,} 個參數")
    print(f"Student 相對 Teacher1 壓縮比例: {t1_params / s_params:.2f}x")
    print(f"Student 相對 Teacher2 壓縮比例: {t2_params / s_params:.2f}x")
    print(f"Student 相對 TA 壓縮比例:       {ta_params / s_params:.2f}x")
    print("=" * 60)

compare_models(teacher1_model, teacher2_model, ta_model, student_model)

# ============================================================
# 7. 多來源知識蒸餾訓練 (distillation.py 對應區塊)
# ============================================================
# 設計說明:
#   Teacher1、Teacher2、TA 三個「已凍結」的模型在同一個 train_step 裡
#   同時對 Student 算 KD loss (各自KLDivergence)，再依權重加總，
#   不是先訓練TA再單獨訓練Student的兩階段做法。
#
#   loss = ALPHA * student_loss
#        + (1 - ALPHA) * (W_T1*KD_t1 + W_T2*KD_t2 + W_TA*KD_ta) * T^2
#
#   W_T1 + W_T2 + W_TA = 1，預設平均分配 (各1/3)，之後可依實驗調整。
class MultiSourceDistillationTrainer(tf.keras.Model):
    def __init__(self, student, teacher1, teacher2, ta,
                 w_teacher1=1/3, w_teacher2=1/3, w_ta=1/3):
        super().__init__()
        self.student = student
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.ta = ta
        self.w_teacher1 = w_teacher1
        self.w_teacher2 = w_teacher2
        self.w_ta = w_ta

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

        # 三個指導來源預測 (不訓練)
        t1_preds = self.teacher1(x, training=False)
        t2_preds = self.teacher2(x, training=False)
        ta_preds = self.ta(x, training=False)

        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)

            # Hard loss: 學生 vs 真實標籤
            student_loss = self.student_loss_fn(y, student_preds)

            # Soft loss: 學生 分別 vs 三個來源 (用溫度軟化後算 KLDivergence)
            student_soft = tf.nn.softmax(student_preds / self.temperature, axis=1)

            kd_t1 = self.distillation_loss_fn(
                tf.nn.softmax(t1_preds / self.temperature, axis=1), student_soft)
            kd_t2 = self.distillation_loss_fn(
                tf.nn.softmax(t2_preds / self.temperature, axis=1), student_soft)
            kd_ta = self.distillation_loss_fn(
                tf.nn.softmax(ta_preds / self.temperature, axis=1), student_soft)

            distillation_loss = (self.w_teacher1 * kd_t1
                                  + self.w_teacher2 * kd_t2
                                  + self.w_ta * kd_ta)

            # 總損失
            loss = (self.alpha * student_loss +
                    (1 - self.alpha) * distillation_loss * (self.temperature ** 2))

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, tf.nn.softmax(student_preds, axis=1))

        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        results['student_loss'] = student_loss
        results['kd_loss'] = distillation_loss
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
W_TEACHER1  = 1 / 3
W_TEACHER2  = 1 / 3
W_TA        = 1 / 3
NUM_EPOCHS  = 50  # 建議 50+，示範可改 10

trainer = MultiSourceDistillationTrainer(
    student=student_model,
    teacher1=teacher1_model, teacher2=teacher2_model, ta=ta_model,
    w_teacher1=W_TEACHER1, w_teacher2=W_TEACHER2, w_ta=W_TA,
)
trainer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    alpha=ALPHA,
    temperature=TEMPERATURE
)

print(f"\n多來源知識蒸餾配置：")
print(f"  溫度參數 (Temperature): {TEMPERATURE}")
print(f"  Alpha (Hard loss 權重): {ALPHA}")
print(f"  三來源權重: Teacher1={W_TEACHER1:.2f}, Teacher2={W_TEACHER2:.2f}, TA={W_TA:.2f}")
print(f"  訓練輪數: {NUM_EPOCHS}")

# Checkpoint 儲存最佳模型 (只存 student 子模型,避免存到整個 Trainer)
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

print("\n--- 開始多來源知識蒸餾訓練 (Teacher1 + Teacher2 + TA 同時指導 Student) ---")
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

# --- Accuracy (Teacher1 / Teacher2 / TA / Student 全部比較) ---
teacher1_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
_, teacher1_acc = teacher1_model.evaluate(test_loader, verbose=0)

teacher2_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
_, teacher2_acc = teacher2_model.evaluate(test_loader, verbose=0)

ta_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
_, ta_acc = ta_model.evaluate(test_loader, verbose=0)

best_student.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
_, student_acc = best_student.evaluate(test_loader, verbose=0)

best_source_acc = max(teacher1_acc, teacher2_acc, ta_acc)

print("=" * 60)
print("最終效能比較")
print("=" * 60)
print(f"Teacher1 (EfficientNetB3) 準確率: {teacher1_acc*100:.2f}%")
print(f"Teacher2 (EfficientNetB0) 準確率: {teacher2_acc*100:.2f}%")
print(f"TA (ResNet20) 準確率:             {ta_acc*100:.2f}%")
print(f"Student (CNN8) 準確率:            {student_acc*100:.2f}%")
print(f"效能保留率 (相對最佳來源): {100 * student_acc / best_source_acc:.2f}%")
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
plt.title('Student Model (CNN8) - Confusion Matrix')
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
plt.title('Student Model (CNN8) - ROC Curve')
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
    f.write("Multi-Source Knowledge Distillation Report\n")
    f.write("Teacher1 (EfficientNetB3) + Teacher2 (EfficientNetB0) + TA (ResNet20) -> Student (CNN8)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Teacher1 Accuracy: {teacher1_acc*100:.2f}%\n")
    f.write(f"Teacher2 Accuracy: {teacher2_acc*100:.2f}%\n")
    f.write(f"TA Accuracy:       {ta_acc*100:.2f}%\n")
    f.write(f"Student Accuracy:  {student_acc*100:.2f}%\n")
    f.write(f"Performance Retention (vs best source): {100 * student_acc / best_source_acc:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    for line in report_lines:
        f.write(line + "\n")

print(f"✓ 完整報告已儲存至: {report_path}")
print("\n=== 所有任務完成！===")