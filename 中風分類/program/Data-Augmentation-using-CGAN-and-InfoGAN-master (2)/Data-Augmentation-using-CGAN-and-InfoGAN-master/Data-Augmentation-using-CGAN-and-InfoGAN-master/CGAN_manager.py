import time
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import os # <-- 新增匯入 os 模組

import utils
# 假設您已經將 CGAN_model_MNIST.py 複製並修改為 CGAN_model_CT.py
from CGAN_model_CT import Generator, Discriminator, get_noise
from CONSTANTS import Constants


class CGANManager:
    def __init__(self, device, dataset_name, dataloader, n_classes, z_dim):
        self.device = device
        self.dataset_name = dataset_name
        self.dataloader = dataloader
        self.n_classes = n_classes
        self.z_dim = z_dim

    def get_one_hot_labels(self, labels, n_classes):
        """
        Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
        """
        return F.one_hot(labels, n_classes)

    def combine_vectors(self, x, y):
        """
        Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
        """
        # Note: Make sure this function outputs a float no matter what inputs it receives
        combined = torch.cat((x.float(), y.float()), 1)
        return combined

    def get_input_dimensions(self, z_dim, image_shape, n_classes):
        """
        Function for getting the size of the conditional input dimensions
        from z_dim, the image shape, and number of classes.
        """
        generator_input_dim = z_dim + n_classes
        discriminator_im_chan = image_shape[0] + n_classes
        return generator_input_dim, discriminator_im_chan

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def train_CGAN(self):
        # 定義您的 CT 影像尺寸
        image_shape = (1, 128, 128) # (通道數, 高度, 寬度) - 您可以根據需要調整
        generator_input_dim, discriminator_im_chan = self.get_input_dimensions(self.z_dim, image_shape,
                                                                               self.n_classes)

        # 確保您已經修改了 CGAN_model_CT.py 中的網路架構以適應 128x128
        gen_input_dim = self.z_dim + self.n_classes
        gen = Generator(input_dim=gen_input_dim, im_chan=image_shape[0], hidden_dim=128).to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=Constants.C_GAN_LR_G, betas=(0.5, 0.999))
        
        disc_input_chan = image_shape[0] + self.n_classes
        disc = Discriminator(im_chan=disc_input_chan, hidden_dim=128).to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=Constants.C_GAN_LR_D, betas=(0.5, 0.999))

        gen = gen.apply(self.weights_init)
        disc = disc.apply(self.weights_init)

        generator_losses = []            # 逐步(batch)紀錄
        discriminator_losses = []        # 逐步(batch)紀錄
        epoch_gen_losses = []            # 每個 epoch 的平均
        epoch_disc_losses = []           # 每個 epoch 的平均
        criterion = nn.BCEWithLogitsLoss()

        start_time = time.time()
        print("Starting Training Loop...")
        for epoch in range(Constants.CGAN_EPOCH):
            epoch_start_time = time.time()
            running_gen_loss = 0.0
            running_disc_loss = 0.0
            running_steps = 0
            for i, (real, labels) in enumerate(self.dataloader):
                cur_batch_size = len(real)
                real = real.to(self.device)

                one_hot_labels = self.get_one_hot_labels(labels.to(self.device), self.n_classes)
                # 將標籤擴展到與 CT 影像相同的尺寸
                image_one_hot_labels = one_hot_labels[:, :, None, None]
                image_one_hot_labels = image_one_hot_labels.repeat(1, 1, image_shape[1], image_shape[2])

                ### 更新判別器 ###
                disc_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)
                noise_and_labels = self.combine_vectors(fake_noise, one_hot_labels)
                fake = gen(noise_and_labels)
                fake_image_and_labels = self.combine_vectors(fake.detach(), image_one_hot_labels)
                real_image_and_labels = self.combine_vectors(real, image_one_hot_labels)
                disc_fake_pred = disc(fake_image_and_labels)
                disc_real_pred = disc(real_image_and_labels)
                # 標籤平滑
                real_targets = torch.full_like(disc_real_pred, Constants.LABEL_SMOOTH_TRUE)
                fake_targets = torch.full_like(disc_fake_pred, Constants.LABEL_SMOOTH_FAKE)

                disc_fake_loss = criterion(disc_fake_pred, fake_targets)
                disc_real_loss = criterion(disc_real_pred, real_targets)
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss.backward()
                disc_opt.step()
                discriminator_losses.append(disc_loss.item())
                running_disc_loss += disc_loss.item()

                ### 更新生成器 ###
                # 生成器可多次更新（每次需重新前向，避免重用已反傳的計算圖）
                g_updates = max(1, int(Constants.CGAN_G_UPDATES_PER_D))
                cur_g_loss = 0.0
                for _ in range(g_updates):
                    gen_opt.zero_grad()
                    # 重新生成 fake（使用相同 noise 與 labels 亦可）
                    fake = gen(noise_and_labels)
                    fake_image_and_labels = self.combine_vectors(fake, image_one_hot_labels)
                    disc_fake_pred = disc(fake_image_and_labels)
                    # 目標希望判別為真（使用 1.0）
                    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                    gen_loss.backward()
                    gen_opt.step()
                    cur_g_loss += gen_loss.item()
                cur_g_loss /= g_updates
                generator_losses.append(cur_g_loss)
                running_gen_loss += gen_loss.item()
                running_steps += 1

                if i != 0 and i % 100 == 0:
                    print(f'[{epoch + 1}/{Constants.CGAN_EPOCH}][{i}/{len(self.dataloader)}]\tLoss_D: {disc_loss.item():.4f}\tLoss_G: {gen_loss.item():.4f}')

            # 紀錄並輸出本 epoch 平均 loss
            avg_g = running_gen_loss / max(1, running_steps)
            avg_d = running_disc_loss / max(1, running_steps)
            epoch_gen_losses.append(avg_g)
            epoch_disc_losses.append(avg_d)

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{Constants.CGAN_EPOCH} | Loss_G(avg): {avg_g:.4f} | Loss_D(avg): {avg_d:.4f} | Time: {epoch_time:.2f}s")

        training_time = time.time() - start_time
        print("-" * 50)
        print(f'Training finished!\nTotal Time for Training: {training_time / 60:.2f}m')
        print("-" * 50)

        # --- 視覺化與紀錄 ---
        try:
            os.makedirs('C_GAN_Images', exist_ok=True)
            plt.figure(figsize=(8,5))
            plt.plot(epoch_gen_losses, label='Gen (avg/epoch)')
            plt.plot(epoch_disc_losses, label='Disc (avg/epoch)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('CGAN Training Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig('C_GAN_Images/Loss.png')
            plt.close()
            # 輸出 CSV
            with open('C_GAN_Images/loss.csv', 'w', encoding='utf-8') as f:
                f.write('epoch,gen_loss,disc_loss\n')
                for idx, (g, d) in enumerate(zip(epoch_gen_losses, epoch_disc_losses), start=1):
                    f.write(f"{idx},{g:.6f},{d:.6f}\n")
        except Exception as e:
            print(f"[WARN] 無法輸出損失曲線或CSV: {e}")

        # --- 修改部分開始：在存檔前建立資料夾 ---
        # 定義儲存路徑
        checkpoint_dir = 'checkpoint/C_GAN'
        # 建立資料夾 (如果不存在)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 儲存模型權重
        save_path = os.path.join(checkpoint_dir, f'C_GAN_FINAL_MODEL_{self.dataset_name}.pt')
        torch.save({
            'Generator': gen.state_dict(),
            'discriminator': disc.state_dict(),
            'optimD': disc_opt.state_dict(),
            'optimG': gen_opt.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")
        # --- 修改部分結束 ---

    # ... (test_CGAN 函數維持不變，您可以之後再修改它來生成您的影像)
    def generate_images(self, num_images_per_class=1000, save_path='./C_GAN_generate_datasets'):
        """
        使用訓練好的CGAN模型生成CT影像
        """
        # 確保目錄存在
        os.makedirs(save_path, exist_ok=True)
        
        # 載入訓練好的模型
        checkpoint_path = os.path.join('checkpoint/C_GAN', f'C_GAN_FINAL_MODEL_{self.dataset_name}.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
            
        # 定義影像形狀
        image_shape = (1, 128, 128)  # CT影像形狀
        gen_input_dim = self.z_dim + self.n_classes
        
        # 建立生成器
        gen = Generator(input_dim=gen_input_dim, im_chan=image_shape[0], hidden_dim=128).to(self.device)
        
        # 載入權重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        gen.load_state_dict(checkpoint['Generator'])
        gen.eval()
        
        print(f"Generating {num_images_per_class} images for each class...")
        
        generated_images = []
        generated_labels = []
        
        with torch.no_grad():
            for class_idx in range(self.n_classes):
                print(f"Generating images for class {class_idx}...")
                
                # 為每個類別生成影像
                for i in range(0, num_images_per_class, 32):  # 批次生成，每批32張
                    batch_size = min(32, num_images_per_class - i)
                    
                    # 產生雜訊
                    fake_noise = get_noise(batch_size, self.z_dim, device=self.device)
                    
                    # 建立one-hot標籤
                    class_labels = torch.full((batch_size,), class_idx, device=self.device)
                    one_hot_labels = self.get_one_hot_labels(class_labels, self.n_classes)
                    
                    # 組合雜訊和標籤
                    noise_and_labels = self.combine_vectors(fake_noise, one_hot_labels)
                    
                    # 生成影像
                    fake_images = gen(noise_and_labels)
                    
                    generated_images.append(fake_images.cpu())
                    generated_labels.append(class_labels.cpu())
        
        # 合併所有生成的影像
        all_generated_images = torch.cat(generated_images, dim=0)
        all_generated_labels = torch.cat(generated_labels, dim=0)
        
        print(f"Generated {len(all_generated_images)} images total")
        print(f"Image shape: {all_generated_images.shape}")
        print(f"Label shape: {all_generated_labels.shape}")
        
        # 儲存生成的資料
        save_file_path = os.path.join(save_path, f'{len(all_generated_images)}k_image_set_{self.dataset_name}_noise_1.pt')
        torch.save(all_generated_images, save_file_path)
        print(f"Generated images saved to {save_file_path}")
        
        return all_generated_images, all_generated_labels

    def test_CGAN(self):
        """測試CGAN並生成樣本影像"""
        try:
            generated_images, generated_labels = self.generate_images(num_images_per_class=100)
            
            # 顯示一些生成的影像
            import matplotlib.pyplot as plt
            
            # 選擇每個類別的幾張影像來顯示
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for i in range(2):  # 2個類別
                for j in range(5):  # 每個類別5張影像
                    # 找到第i個類別的影像
                    class_indices = (generated_labels == i).nonzero(as_tuple=True)[0]
                    if len(class_indices) > j:
                        img_idx = class_indices[j]
                        img = generated_images[img_idx].squeeze()
                        axes[i, j].imshow(img, cmap='gray')
                        axes[i, j].set_title(f'Class {i}')
                        axes[i, j].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'./C_GAN_Images/Generated_CT_Images_{self.dataset_name}.png')
            plt.show()
            
        except Exception as e:
            print(f"Error in test_CGAN: {e}")
            print("Make sure you have trained the model first by calling train_CGAN()")