import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve

def rgb2gray(image):
    """Convert RGB image to grayscale."""
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def sobel_filters(image):
    """Compute Sobel gradients."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Gx = convolve(image, Kx)
    Gy = convolve(image, Ky)

    G = np.hypot(Gx, Gy)  # Gradient magnitude
    theta = np.arctan2(Gy, Gx)  # Gradient direction

    return G, theta

def non_maximum_suppression(gradient, theta):
    """Apply Non-Maximum Suppression."""
    ### APPLY NON MAXIMA SUPRESSION HERE. CONSIDER EDGES TO BE ALONG 0,45,90,135 degrees.
    h, w = gradient.shape
    suppressed = np.zeros_like(gradient)

    angle = theta * (180 / np.pi)   # Convert to degrees
    angle[angle < 0] += 180         # Ensure positive angles

    for i in range(1, h-1):
        for j in range(1, w-1):
            q, r = 255, 255  # Default values for comparison

            # Define neighbors based on the gradient direction
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):  # Horizontal (0°)
                q, r = gradient[i, j+1], gradient[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:  # Diagonal (45°)
                q, r = gradient[i-1, j+1], gradient[i+1, j-1]
            elif 67.5 <= angle[i, j] < 112.5:  # Vertical (90°)
                q, r = gradient[i-1, j], gradient[i+1, j]
            elif 112.5 <= angle[i, j] < 157.5:  # Diagonal (135°)
                q, r = gradient[i-1, j-1], gradient[i+1, j+1]

            # Suppress non-maximum pixels
            if gradient[i, j] >= q and gradient[i, j] >= r:
                suppressed[i, j] = gradient[i, j]
    return suppressed


def threshold(image, low, high):
    """Apply double thresholding."""
    strong = 255
    weak = 75

    strong_i, strong_j = np.where(image >= high)
    weak_i, weak_j = np.where((image <= high) & (image >= low))

    output = np.zeros_like(image, dtype=np.uint8)
    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak

    return output, strong, weak

def hysteresis(image, strong, weak):
    """Apply edge tracking by hysteresis."""
    ### IMPLEMENT HYSTERISIS ON THE IMAGE WITH AN 8 NEIGHBORHOOD
    h, w = image.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if image[i, j] == weak:
                if (image[i-1:i+2, j-1:j+2] == strong).any():
                    image[i, j] = strong
                else:
                    image[i, j] = 0  # Suppress isolated weak edges

    return image

### My image visualisation ###
def show_images(images, names = ['original', 'result']):
  k = len(images)
  plt.figure(figsize=(20, 5))
  for i in range(k):
    img, name = images[i], names[i]
    plt.subplot(1, k, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(name)
    plt.axis("off")
  plt.show()
################################

def canny_edge_detection(image, sigma=1.4, low=20, high=40):
    """Full Canny Edge Detection Pipeline."""
    gray = rgb2gray(image)
    blurred = gaussian_filter(gray, sigma=sigma)
    gradient, theta = sobel_filters(blurred)
    suppressed = non_maximum_suppression(gradient, theta)
    thresholded, strong, weak = threshold(suppressed, low, high)
    final_edges = hysteresis(thresholded, strong, weak)

    # show_images(images=[image, blurred, gradient, suppressed, thresholded, final_edges],
    #             names= ['Original', 'guassian_filtered', 'gradient_magnitude','nonmax_supressed', 'threshold','final_edges'])

    return final_edges

if __name__== "__main__":
# Load an example image
    image = plt.imread('BSR/BSDS500/data/images/train/2092.jpg')
    edges = canny_edge_detection(image)

    show_images([image, edges], ['original', 'result'])