# -*- coding: utf-8 -*-
"""
CT Scan Classification for Stroke Detection
(訓練程式 - 修正模型建構方式、加入數據統計、Bleeding 改為 Hemorrhagic)
"""

# === Main Libraries ===
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
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# === Configuration ===
data_dir = r"C:\Users\USER\Desktop\CT_PRO\original_data"
cgan_dir = r"C:\Users\USER\Desktop\CT_PRO\Generated_Images1\Generated_Images"
output_dir = r"C:\Users\USER\Desktop\CT_PRO\實驗_加入資料增強1015\result"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
print(f"Output directory created at: {output_dir}")

# === Get Paths ===
ischemia_path = os.path.join(data_dir, 'Ischemia', 'PNG')
hemorrhagic_path = os.path.join(data_dir, 'Hemorrhagic', 'PNG')  # ✅ 改這裡
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

for img_list, label_name in [(normal_images, "Normal"), (ischemia_images, "Ischemia"), (hemorrhagic_images, "Hemorrhagic")]:
    path = locals()[f"{label_name.lower()}_path"]
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
sns.countplot(data=df, x='Label', palette='Set2')
plt.title("Distribution of Classes")
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

# === Split the data ===
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Label'])
print(f"\nOriginal Split: Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

# === 加入 CGAN 資料 ===
print("\n--- Merging CGAN data ---")
if os.path.exists(cgan_dir) and os.path.isdir(cgan_dir):
    cgan_imgs = []
    cgan_labels = []
    for class_label in os.listdir(cgan_dir):
        class_path = os.path.join(cgan_dir, class_label)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                cgan_imgs.append(img_path)
                cgan_labels.append(class_label)
    if cgan_imgs:
        cgan_df = pd.DataFrame({'Image_path': cgan_imgs, 'Label': cgan_labels})
        train_df = pd.concat([train_df, cgan_df], ignore_index=True)
        print(f"✅ 合併成功！已將 {len(cgan_df)} 張 CGAN 生成影像加入訓練集。")
        print(f"New total train samples: {len(train_df)}")
    else:
        print("⚠️ CGAN directory is empty.")
else:
    print(f"⚠️ CGAN directory not found at '{cgan_dir}'.")

# === 數據分布統計 ===
print("\n--- Data Distribution After CGAN Augmentation ---")
train_counts_after = train_df['Label'].value_counts()
val_counts = val_df['Label'].value_counts()
test_counts = test_df['Label'].value_counts()

summary_after_df = pd.DataFrame({
    'Training': train_counts_after,
    'Validation': val_counts,
    'Testing': test_counts
}).fillna(0).astype(int)
summary_after_df['Total'] = summary_after_df.sum(axis=1)
summary_after_df.loc['Total'] = summary_after_df.sum()
print(summary_after_df)

# === Generator ===
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='Image_path', y_col='Label',
    target_size=(224, 224), batch_size=32, class_mode='categorical'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='Image_path', y_col='Label',
    target_size=(224, 224), batch_size=32, class_mode='categorical'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='Image_path', y_col='Label',
    target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

# === EfficientNet Model ===
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

base_model.trainable = True
for layer in base_model.layers[:150]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

checkpoint_filepath = os.path.join(output_dir, "best_finetuned_model.keras")
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

print("\n--- Starting Fine-tuning ---")
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[checkpoint_callback]
)
print("\n--- Fine-tuning Finished ---")
print(f"✅ Best model saved to '{checkpoint_filepath}'")

# === Plot and Save acc & loss ===
acc = history_finetune.history['accuracy']
val_acc = history_finetune.history['val_accuracy']
loss = history_finetune.history['loss']
val_loss = history_finetune.history['val_loss']

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

val_loss, val_accuracy = best_model.evaluate(test_generator)
print(f"Evaluation on Test Set: loss is {val_loss:.4f}, accuracy is {val_accuracy*100:.2f}%")

# === Report & Confusion Matrix ===
y_pred_probs = best_model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

report_filepath = os.path.join(output_dir, "classification_report.txt")
with open(report_filepath, 'w') as f:
    f.write(f"Test Set Accuracy: {val_accuracy*100:.2f}%\n")
    f.write(f"Test Set Loss: {val_loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

print(f"Classification report saved to '{report_filepath}'")
print(f"Confusion matrix plot saved to '{os.path.join(output_dir, 'confusion_matrix.png')}'")
print("\n--- All tasks completed and results saved. ---")
