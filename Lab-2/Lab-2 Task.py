import cv2
import numpy as np


def show_image(image): 
    cv2.imshow("Lena" ,  image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread("Lena.jpg" ,0)

show_image(image)
def convolve(image : np.array ,  kernel : np.array) -> np.array:
    kernel = np.flip(kernel)
    ih, iw = image.shape
    kh , kw = kernel.shape
    border_size = kh//2
    img_bordered = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)
    bh,bw = img_bordered.shape
    result = np.zeros((ih, iw), dtype=np.float32)
    for i in range(ih):
        for j in range(iw):
            region = img_bordered[i:i+kh ,j:j+kw]
            mul = np.multiply(region,kernel)
            val = np.sum(mul)
            result[i,j] = val 
    return result

def gaussian_function(u,v , sigma = 1 ):
    return (1 / (2*np.pi * sigma**2))*np.exp(-(u**2 +v**2) / 2 * sigma**2 ) 

def derivative_kernel_x(size , sigma):
    kernel = np.zeros((size,size))
    k = size // 2
    h,w = kernel.shape 
    for i in range(h):
        for j in range(w):
            x = i - k 
            y = j - k
            value_of_gauss = gaussian_function(x, y, sigma)
            kernel[i,j] = -(y/sigma**2) *value_of_gauss 
    kernel = kernel/np.sum(kernel)
    return kernel
def derivative_kernel_y(size , sigma):
    kernel = np.zeros((size,size))
    k = size // 2
    h,w = kernel.shape 
    for i in range(h):
        for j in range(w):
            x = i - k 
            y = j - k
            value_of_gauss = gaussian_function(x, y, sigma)
            kernel[i,j] = -(x/sigma**2) *value_of_gauss 
    kernel = kernel/np.sum(kernel)
    return kernel

kernel_x = derivative_kernel_x(7, 1)
kernel_y = derivative_kernel_y(7, 1)
##kernel_y = np.flip(kernel_x)
print(kernel_x)
print(kernel_y)

derivative_image_x = convolve(image , kernel_x)
derivative_image_y = convolve(image  , kernel_y)

norm_derivative_x = np.round(cv2.normalize(derivative_image_x, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
norm_derivative_y =  np.round(cv2.normalize(derivative_image_x, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
show_image(norm_derivative_x)
show_image(norm_derivative_y)

def mer_image(image1,image2):
    sqr1 = np.square(image1)
    sqr2 = np.square(image2)
    add = sqr1 + sqr2
    return np.sqrt(add)

merged_image = mer_image(derivative_image_x , derivative_image_y)
merged_norm = np.round(cv2.normalize(merged_image, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
show_image(merged_norm)



_, threshold_image = cv2.threshold(merged_norm, 50, 255, cv2.THRESH_BINARY)
# show_image(threshold_image)

def threshold(image , high = 100 , low = 50):
    result = np.zeros_like(image)
    result[ (image >= high) ] = 255
    result[ (image >= low) & (image <= high)] = 128
    return result
th_image = threshold(merged_norm)
show_image(th_image)
    