import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

# Read in the image
#image = mpimg.imread('test.jpg')

#image = mpimg.imread('/home/kaleem/workspace/jupyter/udacity/w1/lanefinding/CarND-LaneLines-P1/test_images/solidYellowCurve2.jpg')
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
#    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
#    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
#    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output


def func(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    thresholds_yellow = (hls[:,:,0] < 23) & (hls[:,:,0] > 19 ) | (hls[:,:,1] > 210 )
    color_select = np.copy(image) *0
    color_select[thresholds_yellow] = [255,255,255]
    img_gray = cv2.cvtColor(color_select, cv2.COLOR_BGR2GRAY)
    
    ksize = 15 # Choose a larger odd number to smooth gradient measurements
    gradx1 = abs_sobel_thresh(img_gray, orient='x', sobel_kernel=ksize, thresh=(70, 255))
    grady1 = abs_sobel_thresh(img_gray, orient='y', sobel_kernel=ksize, thresh=(50, 255))
    mag_binary1 = mag_thresh(img_gray, sobel_kernel=ksize, mag_thresh=(50, 255))
    combined = np.zeros_like(gradx1)
    combined[((gradx1 == 1) & (grady1 == 1)) | ((mag_binary1 == 1)) ] = 1
    

    return combined

#image = mpimg.imread('/home/kaleem/workspace/jupyter/udacity/w1/lanefinding/CarND-LaneLines-P1/test_images/solidWhiteCurve.jpg')
#func(image)
    
images = glob.glob('test_images/*.jpg')
nCount = 0
for idx, fname in enumerate(images):
    image = mpimg.imread(fname)
    
    img1 = func(image)
    
    plt.figure(nCount)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(image)
#    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(img1)
    ax2.set_title('Binary Image', fontsize=30)
    nCount+=1