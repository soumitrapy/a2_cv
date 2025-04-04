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
		self.conv1 = nn.Sequential(*list(vgg.features.children())[0:6])
		self.conv2 = nn.Sequential(*list(vgg.features.children())[6:13])
		self.conv3 = nn.Sequential(*list(vgg.features.children())[13:23])
		self.conv4 = nn.Sequential(*list(vgg.features.children())[23:33])
		self.conv5 = nn.Sequential(*list(vgg.features.children())[33:43])
		self.convs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

		self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
		self.side2 = nn.Sequential(nn.Conv2d(128, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
		self.side3 = nn.Sequential(nn.Conv2d(256, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
		self.side4 = nn.Sequential(nn.Conv2d(512, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
		self.side5 = nn.Sequential(nn.Conv2d(512, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
		self.sides = [self.side1, self.side2, self.side3, self.side4, self.side5]
		assert len(self.sides)==len(self.convs)

		self.ups1 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2)
		self.ups2 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=8,stride=4)
		self.ups3 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=16,stride=8)
		self.ups4 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=32,stride=16)
		self.upsamples = [None, self.ups1, self.ups2, self.ups3, self.ups4]
		assert len(self.upsamples)==len(self.convs)

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
			side_out = F.sigmoid(self.crop(side_out, h, w))
			outputs.append(side_out)
			x = conv_out
		yfuse = self.fuse(torch.cat(outputs, axis=1))
		outputs.append(yfuse)
		return outputs

# Class Balanced CrossEntropy Loss
class CBCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, y):
        eps = 1e-20
        beta = torch.sum(1-y, dim = (1,2,3))/torch.sum(torch.ones_like(y), dim = (1,2,3))
        s1 = (y*torch.log(p+eps)).sum(dim=(1,2,3))
        s2 = ((1-y)*torch.log(1-p+eps)).sum(dim=(1,2,3))
        loss = -torch.sum(beta*s1+(1-beta)*s2)/s1.shape[0]
        return loss
        # b = (1-torch.sum(y))/torch.sum(torch.ones_like(y))
        # loss = torch.sum(-b*torch.log(p)*y-(1-b)*torch.log(1-p)*(1-y))
        # torch.log(p)
        # return loss

class HEDLoss(nn.Module):
	def __init__(self, alpha = [1,2,3,4,5]):
		super().__init__()
		self.cbc = CBCLoss()
		self.alpha = alpha
	
	def forward(self, outputs, y):
		lside = 0.0
		for i in range(len(outputs)-1):
			lside += self.alpha[i]*self.cbc(outputs[i],y)
			#print(lside)
		lfuse = F.binary_cross_entropy(outputs[-1], y)
		return lside + (max(self.alpha)+1)*lfuse


if __name__=="__main__":
    from torchvision import transforms
    from preprocessing import load_data
    from torch.optim import Adam


    transform = transforms.ToTensor()
    target_transform = transforms.ToTensor()
    ds, dl = load_data(dataset_root = './BSR/BSDS500',transform=transform, target_transform=target_transform)
    m = HED(pretrained=True)
    optimizer = Adam(m.parameters(), lr=1e-4)
    cbc = CBCLoss()
    hedloss = HEDLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for images, y in dl[0]:
        #print(images.shape, labels.shape)
        pred = m(images)
        print("loss:",hedloss(pred, y))
        break
    # print(len(pred))
    # for x in pred:
    #     print(x.shape)




