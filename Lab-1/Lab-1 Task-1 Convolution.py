import numpy as np
import cv2

# Load grayscale image
img = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# Add border to preserve edge during convolution (2 pixels for 5x5 kernel)
border_size = 2
img_bordered = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)
img_bordered2 = cv2.copyMakeBorder(img, 0 ,0, border_size, border_size, cv2.BORDER_CONSTANT)
# Gaussian-like kernel
kernel1 = np.array([
    [0,  1,  2,  1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0]
], dtype=np.float32)

kernel2 = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
    ] , dtype = np.float32)


def convolution1(img: np.array, kernel: np.array) -> np.array:
    kh, kw = kernel.shape
    h, w = img.shape
    output = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float32)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img[i:i+kh, j:j+kw]
            conv_value = np.sum(region * kernel)
            output[i, j] = conv_value
    return output

def convolution_center_shift(img: np.array, kernel: np.array) -> np.array:
    kh, kw = kernel.shape
    h, w = img.shape
    output = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float32)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img[i:i+kh, j:j+kw]
            conv_value = np.sum(region * kernel)
            output[i, j] = conv_value
    return output


# Perform convolution
img_conv = convolution1(img_bordered,np.flip( kernel1) )
img_conv2 = convolution_center_shift(img_bordered2, kernel2)

# Normalize the float result to 0–255 and convert to uint8
norm = np.round(cv2.normalize(img_conv, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
norm2 = np.round(cv2.normalize(img_conv2, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)


# Crop to original size (optional: norm should already be h x w)
norm_cropped = norm[:h, :w]
norm_cropped2 = norm[:h, :w]

# Show all images
cv2.imshow('Original Grayscale Image', img)
cv2.imshow('Bordered Image', img_bordered)
cv2.imshow('Convolution Image', img_conv)  
cv2.imshow('Center shifted' ,img_conv2)
cv2.imshow('Normalized Image', norm)
cv2.imshow('Normalized Cropped Image', norm_cropped)
cv2.imshow('Normalized Image Center Shift', norm2)
cv2.imshow('Normalized Cropped Image Center Shift', norm_cropped2)

cv2.waitKey(0)
cv2.destroyAllWindows()
