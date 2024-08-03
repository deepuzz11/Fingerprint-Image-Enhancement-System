import cv2
import numpy as np
from scipy import ndimage

def ridge_segment(img, block_size=16, sigma=0.5):
    """
    Segments ridges from the fingerprint image using local variance thresholding.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    bin_img = np.zeros_like(img)
    rows, cols = img.shape

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = img[i:i+block_size, j:j+block_size]
            std_dev = np.std(block)
            bin_img[i:i+block_size, j:j+block_size] = (block > sigma * std_dev)

    return bin_img

def ridge_orientation(img, block_size=16, ksize=3):
    """
    Estimates ridge orientations in the fingerprint image using gradient-based method.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    orientations = np.zeros((rows, cols))

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = img[i:i+block_size, j:j+block_size]
            sobelx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=ksize)
            theta = np.arctan2(sobely, sobelx)
            orientations[i:i+block_size, j:j+block_size] = np.rad2deg(theta) % 180

    return orientations

def enhance_fingerprint(img):
    """
    Enhances the fingerprint image.
    """
    # Segment ridges
    segmented_img = ridge_segment(img)

    # Estimate ridge orientation
    orientations = ridge_orientation(img)

    # Gabor filtering
    gabor_filter_bank = cv2.getGaborKernel((21, 21), 5, np.pi/2, 10, 0.5, 0, ktype=cv2.CV_32F)
    enhanced_img = np.zeros_like(segmented_img, dtype=np.float32)

    for angle in range(0, 180, 15):
        theta = np.deg2rad(angle)
        for freq in range(4, 9):
            kernel = cv2.getGaborKernel((21, 21), 5, theta, freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered_img = cv2.filter2D(segmented_img, cv2.CV_32F, kernel)
            enhanced_img = np.maximum(enhanced_img, filtered_img)

    return enhanced_img

def binarize(img, threshold=127):
    """
    Binarizes the fingerprint image using a given threshold.
    """
    _, binarized_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binarized_img

def thinning(img):
    """
    Applies morphological thinning to the fingerprint image.
    """
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

# Load fingerprint image
img = cv2.imread('fingerprint.jpg')

# Enhance fingerprint
enhanced_img = enhance_fingerprint(img)

# Binarize the enhanced image
binarized_img = binarize(enhanced_img)

# Apply thinning to the binarized image
thinned_img = thinning(binarized_img)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Enhanced Image', enhanced_img)
cv2.imshow('Binarized Image', binarized_img)
cv2.imshow('Thinned Image', thinned_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
