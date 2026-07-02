import torch, os
from torchvision.utils import save_image

pt_path = r"C:\Users\user\Desktop\Data-Augmentation-using-CGAN-and-InfoGAN-master (2)\C_GAN_generate_datasets\2000k_image_set_stroke_ct_noise_1.pt"
out_dir = r"C:\Users\user\Desktop\cgan結果\export_png"
os.makedirs(out_dir, exist_ok=True)

t = torch.load(pt_path)            # 形狀 [N,1,128,128]，值域約[-1,1]
t = (t.clamp(-1,1) + 1) / 2.0      # 還原到[0,1]
for i, img in enumerate(t):
      save_image(img, os.path.join(out_dir, f"{i:06d}.png"))