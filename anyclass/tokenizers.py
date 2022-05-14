import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# adapted from lucidrains/vit-pytorch
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ImageTokenizer(nn.Module):
    # input (batch_size, channels, height, width)
    # output (batch_size, (image_size / patch_size)**2, dim)
    def __init__(self, *, image_size, patch_size, dim, channels = 3):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return x

class SequenceTokenizer(nn.Module):
    # input (batch_size, channels, input_dim)   # this can be used for one hot vectors as well as 1d sequences
    # output (batch_size, (input_dim/patch_size)+1, dim)
    def __init__(self, input_dim, patch_size, dim, channels=1):
        super().__init__()
        num_patches = input_dim // patch_size
        patch_dim = channels * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d p1) -> b d (p1 c)', p1=patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return x