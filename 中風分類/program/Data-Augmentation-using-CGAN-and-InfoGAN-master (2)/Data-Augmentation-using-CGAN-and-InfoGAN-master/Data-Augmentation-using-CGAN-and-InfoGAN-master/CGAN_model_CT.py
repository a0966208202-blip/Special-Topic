import torch
import torch.nn as nn

# --- 修改部分：這是為 128x128 CT 影像設計的新模型架構 ---

class Generator(nn.Module):
    '''
    Generator Class for 128x128 CT Scans
    Values:
        input_dim: the dimension of the input vector (noise + labels)
        im_chan: the number of channels in the images (1 for grayscale)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=100, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            # Input: input_dim x 1 x 1
            self.make_gen_block(input_dim, hidden_dim * 16, kernel_size=4, stride=1, padding=0), # Output: 1024 x 4 x 4
            self.make_gen_block(hidden_dim * 16, hidden_dim * 8), # Output: 512 x 8 x 8
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4), # Output: 256 x 16 x 16
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2), # Output: 128 x 32 x 32
            self.make_gen_block(hidden_dim * 2, hidden_dim),     # Output: 64 x 64 x 64
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, padding=1, final_layer=True), # Output: 1 x 128 x 128
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Tanh(), # Tanh activation to scale output to [-1, 1]
            )

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        """
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)


class Discriminator(nn.Module):
    '''
    Discriminator Class for 128x128 CT Scans
    Values:
      im_chan: the number of channels in the images (image channels + label channels)
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: im_chan x 128 x 128
            self.make_disc_block(im_chan, hidden_dim), # Output: 64 x 64 x 64
            self.make_disc_block(hidden_dim, hidden_dim * 2), # Output: 128 x 32 x 32
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4), # Output: 256 x 16 x 16
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8), # Output: 512 x 8 x 8
            self.make_disc_block(hidden_dim * 8, hidden_dim * 16), # Output: 1024 x 4 x 4
            self.make_disc_block(hidden_dim * 16, 1, kernel_size=4, stride=1, padding=0, final_layer=True), # Output: 1 x 1 x 1
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

# --- get_noise 函數維持不變 ---
def get_noise(n_samples, input_dim, device='cpu'):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    """
    return torch.randn(n_samples, input_dim, device=device)