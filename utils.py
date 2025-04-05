import matplotlib.pyplot as plt
from datetime import datetime
import torch
import os

def show_images(images, figname="predictions/SimpleCNN_predictions.jpg"):
    n, k = len(images), len(images[0])
    fig, axes = plt.subplots(n, k, figsize=(10, 50))
    
    for i, imgs in enumerate(images):
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()

def save_model(model, dest):
	device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
	model.to('cpu')
	path = dest+type(model).__name__+device.type+str(datetime.now())[:15]+'.pth'
	torch.save(model.state_dict(), path)
	print(f"model saved to {path}")
	model.to(device)
	return path


