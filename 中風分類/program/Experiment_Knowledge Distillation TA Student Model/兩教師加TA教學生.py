# -*- coding: utf-8 -*-
"""
CT Scan Classification - Two-Stage, Three-Source Knowledge Distillation
Stage 1: Teacher1 (EfficientNetB3) + Teacher2 (EfficientNetB0) -> TA (ResNet20)
Stage 2: Teacher1 + Teacher2 + TA (三個來源同時)         -> Student (CNN8)
任務: 3分類 (Normal / Ischemia / Hemorrhagic)
更改CNN8的架構、新增信賴區間

"""

# ============================================================
# 1. 環境設定
# ============================================================

import os
import math
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
OUTPUT_DIR    = r"teacher ta_distillation_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ 輸出資料夾: {OUTPUT_DIR}")

# ============================================================
# 4. 資料載入 (data.py 對應區塊)
# ============================================================
# 沿用 doc2 的作法：同一張圖一次讀進來，resize 成三種尺寸
# (teacher1 / teacher2 / student-size)，確保三者對應同一張圖、同一個 label。
# TA 和 Student 兩個模型都吃 student_size 那份。

TEACHER1_SIZE = (300, 300)   # EfficientNetB3
TEACHER2_SIZE = (224, 224)   # EfficientNetB0，換成你實際的 teacher2 解析度
STUDENT_SIZE  = (300, 300)   # TA (ResNet20) / Student (CNN8) 共用的尺寸

def get_ct_dataloaders(data_dir, batch_size=32,
                        teacher1_size=TEACHER1_SIZE,
                        teacher2_size=TEACHER2_SIZE,
                        student_size=STUDENT_SIZE):
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

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Label'])
    train_df, val_df  = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['Label'])

    print(f"\n資料分布：Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    class_names = sorted(df['Label'].unique())
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    def make_dataset(sub_df, shuffle):
        paths = sub_df['Image_path'].values
        label_idx = sub_df['Label'].map(label_to_idx).values

        def load_and_preprocess(path, label):
            img_raw = tf.io.read_file(path)
            img = tf.image.decode_png(img_raw, channels=3)
            img = tf.cast(img, tf.float32)

            img_t1 = preprocess_input(tf.image.resize(img, teacher1_size))
            img_t2 = preprocess_input(tf.image.resize(img, teacher2_size))
            img_s  = preprocess_input(tf.image.resize(img, student_size))

            label_onehot = tf.one_hot(label, depth=len(class_names))
            return (img_t1, img_t2, img_s), label_onehot

        ds = tf.data.Dataset.from_tensor_slices((paths, label_idx))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=42)
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(train_df, shuffle=True)
    val_ds   = make_dataset(val_df,   shuffle=False)
    test_ds  = make_dataset(test_df,  shuffle=False)

    test_labels = test_df['Label'].map(label_to_idx).values

    return train_ds, val_ds, test_ds, len(train_df), class_names, test_labels

train_loader, val_loader, test_loader, train_size, CLASS_NAMES, test_labels = get_ct_dataloaders(
    data_dir=DATA_DIR, batch_size=32)

num_classes = len(CLASS_NAMES)
print(f"類別數: {num_classes}")
print(f"訓練樣本: {train_size}")
print(f"類別名稱: {CLASS_NAMES}")

# ============================================================
# 5. 模型定義 (models.py 對應區塊)
# ============================================================

def get_teacher_model(model_path):
    model = tf.keras.models.load_model(model_path)
    model.trainable = False
    print(f"✓ Teacher 模型載入完成: {model_path}")
    return model

def resnet_block(x, filters, kernel_size=3, stride=1, use_shortcut=False):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, strides=1,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if use_shortcut or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def get_ta_model(input_shape=STUDENT_SIZE + (3,), num_classes=3):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    x = resnet_block(x, 16, use_shortcut=True)
    x = resnet_block(x, 16)
    x = resnet_block(x, 16)

    x = resnet_block(x, 32, stride=2, use_shortcut=True)
    x = resnet_block(x, 32)
    x = resnet_block(x, 32)

    x = resnet_block(x, 64, stride=2, use_shortcut=True)
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)
    return models.Model(inputs, outputs, name='ResNet20_TA')

def get_student_model(input_shape=STUDENT_SIZE + (3,), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)  

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)
    return models.Model(inputs, outputs, name='CNN8_Student')

teacher1_model = get_teacher_model(TEACHER1_PATH)
teacher2_model = get_teacher_model(TEACHER2_PATH)
ta_model       = get_ta_model(num_classes=num_classes)
student_model  = get_student_model(num_classes=num_classes)

# ============================================================
# 6. 模型比較
# ============================================================
def compare_models(named_models):
    print("=" * 60)
    print("模型參數比較")
    print("=" * 60)
    base_params = None
    for name, model in named_models.items():
        p = model.count_params()
        if base_params is None:
            base_params = p
        print(f"{name}: {p:,} 個參數 ({100*p/base_params:.2f}% of {list(named_models.keys())[0]})")
    print("=" * 60)

compare_models({
    "Teacher1 (EfficientNetB3)": teacher1_model,
    "Teacher2 (EfficientNetB0)": teacher2_model,
    "TA (ResNet20)": ta_model,
    "Student (CNN8)": student_model,
})

# ============================================================
# 7. 知識蒸餾 Trainer 定義 (distillation.py 對應區塊)
# ============================================================

# --- Stage 1: 2 個 Teacher -> TA (與 doc2 相同，不變) ---
class MultiTeacherDistillationTrainer(tf.keras.Model):
    def __init__(self, student, teachers, teacher_weights=None):
        super().__init__()
        self.student = student
        self.teachers = teachers
        n = len(teachers)
        self.teacher_weights = teacher_weights or [1.0 / n] * n
        # 自己管理 metric，不靠 compiled_metrics
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

    def compile(self, optimizer, alpha, temperature):
        super().compile(optimizer=optimizer, run_eagerly=True)  # 拿掉 metrics=[...]
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()

    @property
    def metrics(self):
        # 讓 Keras 每個 epoch 開始時自動幫你 reset_state()
        return [self.acc_metric]

    def _ensemble_teacher_soft(self, x_t1, x_t2):
        soft = 0.0
        inputs_per_teacher = [x_t1, x_t2]
        for w, teacher, x_i in zip(self.teacher_weights, self.teachers, inputs_per_teacher):
            teacher_preds = teacher(x_i, training=False)
            soft += w * tf.nn.softmax(teacher_preds / self.temperature, axis=1)
        return soft

    def train_step(self, data):
        (x_t1, x_t2, x_student), y = data
        teacher_soft = self._ensemble_teacher_soft(x_t1, x_t2)

        with tf.GradientTape() as tape:
            student_preds = self.student(x_student, training=True)
            student_loss = self.student_loss_fn(y, student_preds)
            distillation_loss = self.distillation_loss_fn(
                teacher_soft,
                tf.nn.softmax(student_preds / self.temperature, axis=1)
            )
            loss = (self.alpha * student_loss +
                    (1 - self.alpha) * distillation_loss * (self.temperature ** 2))

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        self.acc_metric.update_state(y, tf.nn.softmax(student_preds, axis=1))
        return {"loss": loss, "accuracy": self.acc_metric.result()}

    def test_step(self, data):
        (x_t1, x_t2, x_student), y = data
        student_preds = self.student(x_student, training=False)
        student_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y, student_preds)

        self.acc_metric.update_state(y, tf.nn.softmax(student_preds, axis=1))
        return {"loss": student_loss, "accuracy": self.acc_metric.result()}

    def get_config(self):
        return {}


# --- Stage 2 (改版重點): Teacher1 + Teacher2 + TA 三個來源同時 -> Student ---
# 沿用 doc1 的 loss 設計 (三路 KD 各自算再加權加總)，
# 但改用 doc2 的三解析度 data pipeline: teacher1 吃 x_t1、teacher2 吃 x_t2、
# TA 和 Student 都吃 x_student (因為 TA 訓練時就是用 student_size)。
class ThreeSourceDistillationTrainer(tf.keras.Model):
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
        # 自己管理 metric，不靠 compiled_metrics
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

    def compile(self, optimizer, alpha, temperature):
        super().compile(optimizer=optimizer, run_eagerly=True)  # 拿掉 metrics=[...]
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()

    @property
    def metrics(self):
        # 讓 Keras 每個 epoch 開始自動幫你 reset_state()
        return [self.acc_metric]

    def train_step(self, data):
        (x_t1, x_t2, x_student), y = data

        t1_preds = self.teacher1(x_t1, training=False)
        t2_preds = self.teacher2(x_t2, training=False)
        ta_preds = self.ta(x_student, training=False)   # TA 吃 student_size

        with tf.GradientTape() as tape:
            student_preds = self.student(x_student, training=True)

            student_loss = self.student_loss_fn(y, student_preds)
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

            loss = (self.alpha * student_loss +
                    (1 - self.alpha) * distillation_loss * (self.temperature ** 2))

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        self.acc_metric.update_state(y, tf.nn.softmax(student_preds, axis=1))

        return {
            "loss": loss,
            "accuracy": self.acc_metric.result(),
            "student_loss": student_loss,
            "kd_loss": distillation_loss,
        }

    def test_step(self, data):
        (_x_t1, _x_t2, x_student), y = data
        student_preds = self.student(x_student, training=False)
        student_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y, student_preds)

        self.acc_metric.update_state(y, tf.nn.softmax(student_preds, axis=1))

        return {
            "loss": student_loss,
            "accuracy": self.acc_metric.result(),
        }

    def get_config(self):
        return {}


# ============================================================
# 7b. Callback：存「內部真正的模型」權重，不是整個 Trainer 外殼
# ============================================================
class SaveBestInnerModelWeights(tf.keras.callbacks.Callback):
    def __init__(self, inner_model, filepath, monitor='val_loss', mode='min'):
        super().__init__()
        self.inner_model = inner_model
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        improved = (current < self.best) if self.mode == 'min' else (current > self.best)
        if improved:
            print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to "
                  f"{current:.5f}, saving model to {self.filepath}")
            self.best = current
            self.inner_model.save_weights(self.filepath)
        else:
            print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}")


# ============================================================
# 8. Stage 1 訓練: 2 個 Teacher -> TA
# ============================================================
TEMPERATURE = 3.0
ALPHA       = 0.7
NUM_EPOCHS  = 10  # 建議 50+，示範可改 10

# Stage 1 用: 兩個 teacher 各自的權重 (可依各自準確率調整)
STAGE1_W_TEACHER1 = 0.6
STAGE1_W_TEACHER2 = 0.4

# Stage 2 用: 三個來源的權重 — TA 給比較高的權重
# (TAKD 的核心論點：TA 容量介於中間，訊號比大 Teacher 更貼近 Student 能吸收的範圍)
STAGE2_W_TEACHER1 = 0.25
STAGE2_W_TEACHER2 = 0.25
STAGE2_W_TA       = 0.50

ta_trainer = MultiTeacherDistillationTrainer(
    student=ta_model,
    teachers=[teacher1_model, teacher2_model],
    teacher_weights=[STAGE1_W_TEACHER1, STAGE1_W_TEACHER2],
)
ta_trainer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    alpha=ALPHA,
    temperature=TEMPERATURE
)

print(f"\nStage 1 知識蒸餾配置 (Teacher x2 -> TA)：")
print(f"  溫度參數 (Temperature): {TEMPERATURE}")
print(f"  Alpha (Hard loss 權重): {ALPHA}")
print(f"  Teacher 權重: Teacher1={STAGE1_W_TEACHER1}, Teacher2={STAGE1_W_TEACHER2}")
print(f"  訓練輪數: {NUM_EPOCHS}")

ta_checkpoint_cb = SaveBestInnerModelWeights(
    inner_model=ta_model,
    filepath=os.path.join(OUTPUT_DIR, 'best_ta.weights.h5'),
    monitor='val_loss',
    mode='min',
)

print("\n--- Stage 1: 開始訓練 TA (ResNet20) ---")
ta_history = ta_trainer.fit(
    train_loader,
    validation_data=val_loader,
    epochs=NUM_EPOCHS,
    callbacks=[ta_checkpoint_cb],
    verbose=1
)
print("\n--- Stage 1 訓練完成 ---")

ta_model.load_weights(os.path.join(OUTPUT_DIR, 'best_ta.weights.h5'))
ta_model.save(os.path.join(OUTPUT_DIR, 'best_ta_model.keras'))
print("✓ TA 模型已儲存！")

# ============================================================
# 9. Stage 2 訓練: Teacher1 + Teacher2 + TA (三來源) -> Student (CNN8)
# ============================================================
print("\n--- 載入最佳 TA 模型，作為 Stage 2 的其中一個指導來源 ---")
best_ta_path = os.path.join(OUTPUT_DIR, 'best_ta_model.keras')
best_ta_model = tf.keras.models.load_model(best_ta_path)
best_ta_model.trainable = False
print(f"✓ 已載入最佳 TA 模型: {best_ta_path}")

student_trainer = ThreeSourceDistillationTrainer(
    student=student_model,
    teacher1=teacher1_model,
    teacher2=teacher2_model,
    ta=best_ta_model,
    w_teacher1=STAGE2_W_TEACHER1,
    w_teacher2=STAGE2_W_TEACHER2,
    w_ta=STAGE2_W_TA,
)
student_trainer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    alpha=ALPHA,
    temperature=TEMPERATURE
)

print(f"\nStage 2 知識蒸餾配置 (Teacher1 + Teacher2 + TA -> Student)：")
print(f"  溫度參數 (Temperature): {TEMPERATURE}")
print(f"  Alpha (Hard loss 權重): {ALPHA}")
print(f"  三來源權重: Teacher1={STAGE2_W_TEACHER1}, Teacher2={STAGE2_W_TEACHER2}, TA={STAGE2_W_TA}")
print(f"  訓練輪數: {NUM_EPOCHS}")

student_checkpoint_cb = SaveBestInnerModelWeights(
    inner_model=student_model,
    filepath=os.path.join(OUTPUT_DIR, 'best_student.weights.h5'),
    monitor='val_loss',
    mode='min',
)

print("\n--- Stage 2: 開始訓練 Student (CNN8) ---")
student_history = student_trainer.fit(
    train_loader,
    validation_data=val_loader,
    epochs=NUM_EPOCHS,
    callbacks=[student_checkpoint_cb],
    verbose=1
)
print("\n--- Stage 2 訓練完成 ---")

student_model.load_weights(os.path.join(OUTPUT_DIR, 'best_student.weights.h5'))
student_model.save(os.path.join(OUTPUT_DIR, 'best_student_model.keras'))
print("✓ 學生模型已儲存！")

# ============================================================
# 10. 繪製訓練曲線
# ============================================================
def plot_training_curves(history, save_dir, tag):
    acc      = history.history['accuracy']
    val_acc  = history.history.get('val_accuracy')
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    if val_acc is not None:
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title(f'{tag} - Training and Validation Accuracy')
    else:
        print(f"⚠ {tag}: history 裡沒有 val_accuracy，只畫 Train Accuracy")
        plt.title(f'{tag} - Training Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title(f'{tag} - Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_{tag}.png'))
    plt.close()
    print(f"✓ {tag} 訓練曲線已儲存")

plot_training_curves(ta_history, OUTPUT_DIR, tag='stage1_TA')
plot_training_curves(student_history, OUTPUT_DIR, tag='stage2_Student')

# ============================================================
# 11. 完整評估 (含 Specificity, NPV, ROC-AUC) — 針對最終 Student (CNN8)
# ============================================================
print("\n--- 載入最佳學生模型進行評估 ---")
best_student_path = os.path.join(OUTPUT_DIR, 'best_student_model.keras')
best_student = tf.keras.models.load_model(best_student_path)
print(f"✓ 已載入最佳學生模型: {best_student_path}")

test_loader_t1      = test_loader.map(lambda xs, y: (xs[0], y))
test_loader_t2      = test_loader.map(lambda xs, y: (xs[1], y))
test_loader_student = test_loader.map(lambda xs, y: (xs[2], y))

y_true = test_labels
n_classes = len(CLASS_NAMES)

y_pred_logits = best_student.predict(test_loader_student.map(lambda x, y: x))
y_pred_probs  = tf.nn.softmax(y_pred_logits, axis=1).numpy()
y_pred        = np.argmax(y_pred_probs, axis=1)

# --- Accuracy: Teacher1 / Teacher2 / TA / Student ---
teacher1_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
_, teacher1_acc = teacher1_model.evaluate(test_loader_t1, verbose=0)

teacher2_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
_, teacher2_acc = teacher2_model.evaluate(test_loader_t2, verbose=0)

best_ta_model.compile(optimizer='adam',
                       loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
_, ta_acc = best_ta_model.evaluate(test_loader_student, verbose=0)

best_student.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
_, student_acc = best_student.evaluate(test_loader_student, verbose=0)

best_source_acc = max(teacher1_acc, teacher2_acc, ta_acc)

print("=" * 60)
print("最終效能比較")
print("=" * 60)
print(f"Teacher1 (EfficientNetB3) 準確率: {teacher1_acc*100:.2f}%")
print(f"Teacher2 (EfficientNetB0) 準確率: {teacher2_acc*100:.2f}%")
print(f"TA (ResNet20) 準確率:             {ta_acc*100:.2f}%")
print(f"Student (CNN8) 準確率:            {student_acc*100:.2f}%")
print(f"效能保留率 (相對三來源中最佳者): {100 * student_acc / best_source_acc:.2f}%")
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

# --- Confidence Interval ---
n = np.sum(cm)
correct_predictions = np.trace(cm)
accuracy = correct_predictions / n
Z = 1.96 
se = math.sqrt((accuracy * (1 - accuracy)) / n)
ci_lower = accuracy - (Z * se)
ci_upper = accuracy + (Z * se)
ci_lower = max(0.0, ci_lower)
ci_upper = min(1.0, ci_upper)

print(" Confidence Interval (95% CI)")
print("="*45)
print(f"▸ 測試總樣本數 (n) : {n}")
print(f"▸ 預測正確數量     : {correct_predictions}")
print(f"▸ 模型準確率 (Acc) : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"▸ 95% 信賴區間     : [{ci_lower:.4f}, {ci_upper:.4f}]")
print("-" * 45)

# --- Specificity & NPV ---
print("\n--- Specificity & NPV (per-class) ---")
specificities, npvs = [], []
report_lines = ["\n--- Additional Metrics (One-vs-Rest) ---"]

# 新增：把信賴區間寫進報告
report_lines.append("\n--- 95% Confidence Interval ---")
report_lines.append(f"測試總樣本數 (n): {n}")
report_lines.append(f"預測正確數量: {correct_predictions}")
report_lines.append(f"模型準確率 (Acc): {accuracy:.4f} ({accuracy*100:.2f}%)")
report_lines.append(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

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
# 12. 儲存完整報告
# ============================================================
report_path = os.path.join(OUTPUT_DIR, 'distillation_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("Two-Stage, Three-Source Knowledge Distillation Report\n")
    f.write("Stage 1: Teacher x2 (EfficientNet) -> TA (ResNet20)\n")
    f.write("Stage 2: Teacher1 + Teacher2 + TA  -> Student (CNN8)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Teacher1 Accuracy: {teacher1_acc*100:.2f}%\n")
    f.write(f"Teacher2 Accuracy: {teacher2_acc*100:.2f}%\n")
    f.write(f"TA Accuracy:       {ta_acc*100:.2f}%\n")
    f.write(f"Student Accuracy:  {student_acc*100:.2f}%\n")
    f.write(f"Performance Retention (vs best of 3 sources): {100 * student_acc / best_source_acc:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    for line in report_lines:
        f.write(line + "\n")

print(f"✓ 完整報告已儲存至: {report_path}")
print("\n=== 所有任務完成！===")