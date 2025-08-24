def laplacian_of_gaussian(image, sigma=1.0):
    gray_image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    kernel = imlib.Gaussian_Sharpenning_kernel(7,sigma)
    log_img = imlib.convolve(gray_image,kernel)
    return log_img

def zero_crossing(log_img):
    zc = np.zeros(log_img.shape, dtype=np.uint8)
    h,w = log_img.shape
    for i in range(1, h-2):
        for j in range(1, w-2):
            patch = log_img[i-1:i+2, j-1:j+2]
            local_std = np.std(patch)
            if np.max(patch) > 0 and np.min(patch) < 0:
                zc[i, j] = np.max(patch)
                
    return zc
def local_variance2(image):
    variance = np.zeros_like(image, dtype=np.float32)

    h,w = image.shape 
    for i in range (1,h-2):
        for j in range(1,w-2):
            region = image[i-1:i+2, j-1:j+2]
            std = np.std(region)
            variance[i,j] = std**2
    return variance
def local_variance(image, ksize=5):
    mean = cv2.blur(image, (ksize, ksize))
    mean_sq = cv2.blur(image**2, (ksize, ksize))
    variance = mean_sq - mean**2
    return variance

def robust_laplacian_edge_detector(image, sigma=1.0, var_thresh=100):
    log_img = laplacian_of_gaussian(image, sigma)
    zc_img = zero_crossing(log_img)
    imlib.show_image(zc_img, 'Zero Crossing Image')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = local_variance2(gray)
    print(variance.shape)
    edge_img = np.zeros_like(zc_img)
    edge_img[(zc_img != 0) & (variance > var_thresh)] = 255
    return edge_img

edge_points = robust_laplacian_edge_detector(image, sigma=1, var_thresh=60)
show_image(edge_points, 'Robust Laplacian Edge Detector')