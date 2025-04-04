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
        sub_nums = [0, 6, 13, 23, 33,43]
        for i in range(len(sub_nums)-1):
            self.convs.append(nn.Sequential(
                *list(vgg.features.children())[sub_nums[i]:sub_nums[i+1]]
            ))
    
    def forward(self,x):
        for m in self.convs:
            out = m(x)
            print(out.shape)
            x = out
        return x



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
    m = HED()
    optimizer = Adam(m.parameters(), lr=1e-4)
    cbc = CBCLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for images, labels in dl[0]:
        print(images.shape, labels.shape)
        pred = m(images)
        print(pred.shape)

        break
