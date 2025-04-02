from preprocessing import BSD500Dataset
from canny import canny_edge_detection
from testing import show_images



data_path = "./BSR/BSDS500"
trainds = BSD500Dataset(data_path, split='train')
testds = BSD500Dataset(data_path, split='test')
valds = BSD500Dataset(data_path, split='val')
#print(len(trainds), len(testds), len(valds))


def canny(sigma=3):
    images = []
    for i in range(10,20):
        x = trainds[i]
        image, label = x['image'], x['gTruth']
        pred = canny_edge_detection(image, sigma=sigma)
        images.append([image, pred, label])
    show_images(images)
canny()


