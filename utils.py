import matplotlib.pyplot as plt
def show_images(images, figname="predictions/SimpleCNN predictions"):
    n, k = len(images), len(images[0])
    fig, axes = plt.subplots(n, k, figsize=(10, 50))
    
    for i, imgs in enumerate(images):
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()