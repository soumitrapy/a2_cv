from preprocessing import BSD500Dataset, load_data
from canny import canny_edge_detection
from utils import show_images
import numpy as np



data_path = "./BSR/BSDS500"
ds, _ = load_data(dataset_root=data_path)
#print(len(trainds), len(testds), len(valds))
trainds = ds[0]

def canny(sigma=3):
    images = []
    for i in range(10,20):
        image, label = trainds[i]
        image, label = np.array(image), np.array(label)
        #print(image.shape, label.shape)
        pred = canny_edge_detection(image, sigma=sigma)
        images.append([image, pred, label])
    show_images(images)
canny()


