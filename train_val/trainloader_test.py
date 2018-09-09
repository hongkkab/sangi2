import torch
import torchvision.transforms as transforms
import sangi_data
import Mytransforms

from skimage import io

train_dir = '../dataset/train/sangi/'

# we dont need randomcrop
train_loader = torch.utils.data.DataLoader(
        sangi_data.Sangi_Data(train_dir, 8), batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

for i, (input, heatmap) in enumerate(train_loader):
    print(input.size(), heatmap.size())
