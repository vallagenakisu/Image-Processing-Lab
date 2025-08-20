import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0  

def convolve2d(image, kernel):
    m, n = kernel.shape
    pad_h, pad_w = m // 2, n // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+m, j:j+n]
            out[i, j] = np.sum(region * kernel)
    return out


def log_kernel(size=7, sigma=1.0):
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    
    for i in range(size):
        for j in range(size):
            u = i - k
            v = j - k
            norm = (u*2 + v2) / (2 * sigma*2)
            kernel[i, j] = (-1.0 / (np.pi * sigma**4)) * (1 - norm) * np.exp(-norm)
    
    return kernel


    

log = log_kernel(size=9, sigma=1.4)
log=np.flip(log)


filtered = convolve2d(img, log)


def zero_crossing(image):
    zc = np.zeros_like(image, dtype=np.uint8)
    rows, cols = image.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            patch = image[i-1:i+2, j-1:j+2]
            p = image[i, j]
            has_negative = False
            has_positive = False
            for u in range(3):
                for v in range(3):
                    val = patch[u, v]
                    if val < 0:
                        has_negative = True
                    if val > 0:
                        has_positive = True

            if (p > 0 and has_negative) or (p < 0 and has_positive):
                zc[i, j] = 1
    return zc

zc_map = zero_crossing(filtered)

def calc_mean(region):
    total = 0.0
    count = region.shape[0] * region.shape[1]
    for u in range(region.shape[0]):
        for v in range(region.shape[1]):
            total += region[u, v]
    mean = total / count
    return mean


def local_variance(img, window=5):
    pad = window // 2
    padded = np.pad(img, pad, mode='constant')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+window, j:j+window]
            mean = calc_mean(region)
            var = calc_mean((region - mean) ** 2)
            out[i, j] = var
    return out

variance_map = local_variance(img, window=7)

th = 0.01
edges = (zc_map == 1) & (variance_map > th)

plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(img, cmap='gray')
plt.subplot(1,3,2); plt.title("Zero Crossings"); plt.imshow(zc_map, cmap='gray')
plt.subplot(1,3,3); plt.title("Final Edges"); plt.imshow(edges, cmap='gray')
plt.show()