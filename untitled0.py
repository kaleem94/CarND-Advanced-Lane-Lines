import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
%matplotlib inline

#GAUSSIAN BLUR PARAMETERS
KERNEL_SIZE = 3

#CANNY EDGE DETECTION PARAMETERS
LOW_THRESHOLD = 75 
HIGH_THRESHOLD = 150 

#REGION OF INTEREST PARAMETERS
H_CONST = 0.08
RIGHT_LANE_SLOPE = 14/32
RIGHT_LANE_CONST = 400
TOP_SHIFT_H = 40 
TOP_SHIFT_V = 40

#HOUGH LINES PARAMETERS
RHO = 3.5 
THETA = np.pi/180
MIN_VOTES = 30    
MIN_LINE_LEN = 5 
MAX_LINE_GAP= 25  
LOWER_EDGE = 800


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


def def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
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


def func_roi(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    thresholds_yellow = (hls[:,:,0] < 21) & (hls[:,:,0] > 19 ) | (hls[:,:,1] > 215 )
    color_select = np.copy(image) *0
    color_select[thresholds_yellow] = [255,255,255]
    img_gray = cv2.cvtColor(color_select, cv2.COLOR_BGR2GRAY)
    
    ksize = 5 # Choose a larger odd number to smooth gradient measurements
    gradx1 = abs_sobel_thresh(img_gray, orient='x', sobel_kernel=ksize, thresh=(70, 255))
    grady1 = abs_sobel_thresh(img_gray, orient='y', sobel_kernel=ksize, thresh=(50, 255))
    mag_binary1 = mag_thresh(img_gray, sobel_kernel=ksize, mag_thresh=(50, 255))
    combined = np.zeros_like(gradx1)
    combined[((gradx1 == 1) & (grady1 == 1)) | ((mag_binary1 == 1)) ] = 1
    
    return combined(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    thresholds_yellow = (hls[:,:,0] < 21) & (hls[:,:,0] > 19 ) | (hls[:,:,1] > 215 )
    color_select = np.copy(image) *0
    color_select[thresholds_yellow] = [255,255,255]
    img_gray = cv2.cvtColor(color_select, cv2.COLOR_BGR2GRAY)
    
    ksize = 5 # Choose a larger odd number to smooth gradient measurements
    gradx1 = abs_sobel_thresh(img_gray, orient='x', sobel_kernel=ksize, thresh=(70, 255))
    grady1 = abs_sobel_thresh(img_gray, orient='y', sobel_kernel=ksize, thresh=(50, 255))
    mag_binary1 = mag_thresh(img_gray, sobel_kernel=ksize, mag_thresh=(50, 255))
    combined = np.zeros_like(gradx1)
    combined[((gradx1 == 1) & (grady1 == 1)) | ((mag_binary1 == 1)) ] = 1
    
    return combined

def get_region_of_interest_vertices(img):
    
    #get image parameters for extracting the region of interest
    img_height = img.shape[0]
    img_width = img.shape[1]

    bottom_left = (img_width/9 - H_CONST*img_width, img_height)
    top_left = (img_width / 2 - (TOP_SHIFT_H ), img_height / 2 + TOP_SHIFT_V)
    top_right = (img_width /2 + TOP_SHIFT_H, img_height/2 + TOP_SHIFT_V)
    bottom_right = (img_width - (RIGHT_LANE_SLOPE*img_width-RIGHT_LANE_CONST) + 
                         (H_CONST*img_width), img_height)
    
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
                        
    #print ('vertices-->', vertices)
    return vertices

def identify_lanes(img):
    
    #gray out the image
    #hsv_img = convert_to_hsv(img)
   
    #smoothen the image
    #smooth_img = gaussian_blur(hsv_img, KERNEL_SIZE)
    
     
    
    #canny edge detection
    #canny_img = canny(mask_img, LOW_THRESHOLD, HIGH_THRESHOLD)
    binary_image = func_roi(img)
    #vertices for extracting desired portion from the image
    vertices = get_region_of_interest_vertices(img)
    
    #poly_img = cv2.polylines(img, vertices, True, (0,255,255),3)
   
    #get portion corresponding to the region of interest from the image
    regions = region_of_interest(binary_image, vertices)
    
    if mode == 'canny':
        return regions
    else:
        #get hough lines for the lanes found in the img
        hough_img = hough_lines(regions, RHO, THETA, MIN_VOTES, MIN_LINE_LEN, MAX_LINE_GAP)

        #return original image masked by the hough lines 
        return weighted_img(hough_img, img)