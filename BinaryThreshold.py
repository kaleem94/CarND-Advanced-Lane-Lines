import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import pickle
import glob

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


def convert_binary(image):
#    image = mpimg.imread('test_images/straight_lines1.jpg')
    #image = mpimg.imread('test_images/test6.jpg')
    # Choose a Sobel kernel size
    ksize = 5 # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(30, 255))
    #grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 255))
    #mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(50, 255))
    #dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.95, 1))
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#    H = hls[:,:,0]
#    L = hls[:,:,1]
    S = hls[:,:,2]
    #plt.figure(2)
    #plt.imshow(S)
    gradx1 = abs_sobel_thresh(S, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    grady1 = abs_sobel_thresh(S, orient='y', sobel_kernel=ksize, thresh=(20, 255))
    mag_binary1 = mag_thresh(S, sobel_kernel=ksize, mag_thresh=(20, 255))
    dir_binary1 = dir_threshold(S, sobel_kernel=ksize, thresh=(1.1, 1.3))
    combined = np.zeros_like(dir_binary1)
    combined[((gradx1 == 1) & (grady1 == 1)) | ((mag_binary1 == 1) & (dir_binary1 == 1))] = 1
#    plt.figure(2)
#    plt.imshow(combined)
    return combined
images = glob.glob('test_images/*.jpg')

# Step through the list and search for chessboard corners
nCount = 0
for idx, fname in enumerate(images):
    img = mpimg.imread(fname)
    retImg = convert_binary(img)
    imgBin = np.zeros_like(img)
    imgGray = np.zeros_like(img)
    imgBin[retImg == 1] = [255,255,255]
    imgGray[retImg == 1] = 255
    
    kernel_size = 5
    blur_masked_edges = np.zeros_like(retImg)
    blur_masked_edges =  cv2.GaussianBlur(imgGray,(kernel_size, kernel_size), 0)
#    plt.figure(2)
#    plt.imshow(blur_masked_edges)
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_length = 30                         
    max_line_gap = 5
    line_image = np.copy(img)*0 #creating a blank to draw lines on
    img_grey = cv2.cvtColor(blur_masked_edges, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(img_grey, rho, theta, threshold,min_line_length, max_line_gap)
#    lines = cv2.HoughLinesP(blur_masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    line_image = np.copy(img)*0
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,255,0),5)
            
#    combo = cv2.addWeighted(img, 0.8, line_image, 1, 0) 
    kernel_size = 5
    low_threshold = 30
    high_threshold = 150
    masked_edges = cv2.Canny(line_image, low_threshold, high_threshold)
    kernel_size = 15
    blur_masked_edges = np.zeros_like(retImg)
    blur_masked_edges =  cv2.GaussianBlur(masked_edges,(kernel_size, kernel_size), 0)
    
#    img_grey = cv2.cvtColor(blur_masked_edges, cv2.COLOR_BGR2GRAY)
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_length = 30                         
    max_line_gap = 30
    lines = cv2.HoughLinesP(blur_masked_edges, rho, theta, threshold,min_line_length, max_line_gap)
    line_image = np.copy(img)*0
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)
            
    combo = cv2.addWeighted(img, 0.8, line_image, 1, 0) 
    
    plt.figure(nCount)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(combo)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(imgBin)
    ax2.set_title('Binary Image', fontsize=30)
    nCount+=1