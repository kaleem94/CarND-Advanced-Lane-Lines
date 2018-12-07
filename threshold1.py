import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import math

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
    ksize = 5 # Choose a larger odd number to smooth gradient measurements
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#    H = hls[:,:,0]
#    L = hls[:,:,1]
    S = hls[:,:,2]
    #plt.figure(2)
    #plt.imshow(S)
    gradx1 = abs_sobel_thresh(S, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    grady1 = abs_sobel_thresh(S, orient='y', sobel_kernel=ksize, thresh=(30, 255))
    mag_binary1 = mag_thresh(S, sobel_kernel=ksize, mag_thresh=(15, 255))
    #dir_binary1 = dir_threshold(S, sobel_kernel=ksize, thresh=(1.49, 1.6))
    combined = np.zeros_like(gradx1)
    #combined[((gradx1 == 1) & (grady1 == 1)) | ((mag_binary1 == 1) & (dir_binary1 == 1))] = 1
    combined[((gradx1 == 1) & (grady1 == 1)) | ((mag_binary1 == 1))] = 1
    kernel_size = 21
    blur_masked_edges = np.zeros_like(combined)
    blur_masked_edges =  cv2.GaussianBlur(combined,(kernel_size, kernel_size), 0)
    #return dir_binary1
    return blur_masked_edges
images = glob.glob('test_images/*.jpg')

def linesDraw(masked_edges,image):
    rho = 5
    theta = np.pi/180
    threshold = 15
    min_line_length = 30                         
    max_line_gap = 40
    line_image = np.copy(image)*0 #creating a blank to draw lines on
    
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    line_image = np.copy(image)*0
#    print (lines)
    for line in lines:
        for x1,y1,x2,y2 in line:
            if(math.sqrt((x1-x2)**2+(y2-y1)**2)>60):
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,255),5)
    
    #color_edges = np.dstack((image, image, image)) 
    
    # Draw the lines on the edge image
    combo = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
    return combo

def hist_quarter(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[(img.shape[0]//2+img.shape[0]//4+img.shape[0]//8+img.shape[0]//16):,:]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

# Create histogram of image binary activations


# Visualize the resulting histogram
#plt.plot(histogram)

# Step through the list and search for chessboard corners
nCount = 0
for idx, fname in enumerate(images):
    img = mpimg.imread(fname)
    retImg = convert_binary(img)
    histogram = hist_quarter(retImg)
    lineImg =linesDraw(retImg,img)
    
    plt.figure(nCount)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.imshow(lineImg)
#    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(retImg)
    ax2.set_title('Binary Image', fontsize=30)
    ax3.set_title('Histogram',fontsize=30)
    ax3.plot(histogram)
    nCount+=1
#    break