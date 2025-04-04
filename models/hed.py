import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class HED(nn.Module):
	def __init__(self, pretrained=False):
		super().__init__()
		weights = VGG16_BN_Weights.DEFAULT if pretrained else None
		vgg = vgg16_bn(weights=weights)
		self.features = vgg.features
		self.convs = []
		sub_nums = [0, 6, 13, 23, 33, 43]
		for i in range(len(sub_nums)-1):
			self.convs.append(nn.Sequential(
				*list(vgg.features.children())[sub_nums[i]:sub_nums[i+1]]
			))
		out_channels = [64, 128, 256, 512, 512]
		self.sides = []
		for i in out_channels:
			self.sides.append(nn.Sequential(
				nn.Conv2d(i, 1, 1),
				nn.BatchNorm2d(1),
				nn.Sigmoid()
			))
		assert len(self.sides)==len(self.convs)
		ups_kernel_sizes = [4,8,16,32]
		assert len(ups_kernel_sizes)==len(self.convs)-1
		self.upsamples = [None]
		for k in ups_kernel_sizes:
			self.upsamples.append(nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=k,stride=k//2))
			
		self.fuse = nn.Sequential(nn.Conv2d(5,1,1), nn.Sigmoid())
	def crop(self, x, h, w):
		xh, xw = x.shape[2:]
		assert xh>=h and xw>=w
		return x[:,:,(xh-h)//2:h+(xh-h)//2, (xw-w)//2:w+(xw-w)//2]
        
        
    
	def forward(self,x):
		h, w = x.shape[2:]
		outputs = []
		for i in range(len(self.convs)):
			conv, side, upsample = self.convs[i], self.sides[i], self.upsamples[i]
			conv_out = conv(x)
			side_out = side(conv_out)
			if upsample:
				side_out=upsample(side_out)
			side_out = self.crop(side_out, h, w)
			outputs.append(side_out)
			x = conv_out
		yfuse = self.fuse(torch.cat(outputs, axis=1))
		outputs.append(yfuse)
		return outputs



if __name__=="__main__":
    from torchvision import transforms
    from preprocessing import load_data
    from models.loss import CBCLoss
    from torch.optim import Adam


    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    target_transform = transforms.ToTensor()
    ds, dl = load_data(dataset_root = './BSR/BSDS500',transform=transform, target_transform=target_transform)
    m = HED(pretrained=True)
    optimizer = Adam(m.parameters(), lr=1e-4)
    cbc = CBCLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for images, labels in dl[0]:
        print(images.shape, labels.shape)
        pred = m(images)
        break
    print(len(pred))
    for x in pred:
        print(x.shape)
