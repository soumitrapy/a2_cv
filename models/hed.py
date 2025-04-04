import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class HED(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        self.vgg = vgg16_bn(weights=weights)
        self.features = self.vgg.features
        self.convs = []
        sub_nums = [0, 6, 13, 23, 33,43]
        for i in range(len(sub_nums)-1):
            self.convs.append(nn.Sequential(
                *list(self.vgg.features.children()[sub_nums[i]:sub_nums[i+1]])
            ))
    
    def forward(self,x):
        pass



if __name__=="__main__":
    from torchvision import transforms
    from preprocessing import load_data
    from loss import CBCLoss
    from torch.optim import Adam


    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    target_transform = transforms.ToTensor()
    ds, dl = load_data(dataset_root = './BSR/BSDS500',transform=transform, target_transform=target_transform)
    m = HED()
    optimizer = Adam(model.parameters(), lr=1e-4)
    cbc = CBCLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for images, labels in dl[0]:
    #     pred = model(images)
    #     print(pred.shape)

    #     break
