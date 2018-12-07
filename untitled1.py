#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
#%matplotlib inline
import os

def extractWhite(image):
    color_select = np.copy(image)
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    thresholds = (image[:,:,0] < rgb_threshold[0]) \
                | (image[:,:,1] < rgb_threshold[1]) \
                | (image[:,:,2] < rgb_threshold[2])
    color_select[thresholds] = [0,0,0]
    return color_select

def extractMajorLaneMarking(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold = 30
    high_threshold = 150
    masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_length = 150                         
    max_line_gap = 18
    line_image = np.copy(image)*0 
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    xHalf = image.shape[1]/2
    yMax = image.shape[0]/2
    x1f=-1
    y1f=-1
    x2f=-1
    y2f=-1
    distThreshold = 0.2
    laneLength = 0
    laneDistFromCenter = math.inf
    slopeM = math.inf
    
#    xd = -1
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                m = (y2-y1)/(x2-x1)
                c = y1-m*x1
                yi = yMax
                if (m!=0):
                    xi = (yi - c)/m
                else:
                    continue
                laneLenLocal = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                dist = (xi - xHalf)
                if (laneLenLocal > laneLength):
                    laneDistFromCenter = dist
                    laneLength = laneLenLocal
                    slopeM = abs(m)
                    x1f = x1
                    x2f = x2
                    y1f = y1
                    y2f = y2
                elif ( slopeM < abs(m) and abs(laneLenLocal - laneLength)< (distThreshold*laneLength) ):
                    laneDistFromCenter = dist
                    laneLength = laneLenLocal
                    slopeM = abs(m)
                    x1f = x1
                    x2f = x2
                    y1f = y1
                    y2f = y2
                elif (slopeM < abs(m)) and dist < laneDistFromCenter:
                    laneDistFromCenter = dist
                    laneLength = laneLenLocal
                    slopeM = abs(m)
                    x1f = x1
                    x2f = x2
                    y1f = y1
                    y2f = y2
                        
    imgWhite1 = extractWhite(image)
    
    kernel_size = 5
    low_threshold = 50
    high_threshold = 150
    masked_edges = cv2.Canny(imgWhite1, low_threshold, high_threshold)
    kernel_size = 5
    blur_masked_edges = cv2.GaussianBlur(masked_edges,(kernel_size, kernel_size), 0)
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_length = 20                         
    max_line_gap = 20    
    linesWhite = cv2.HoughLinesP(blur_masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

    xLine1 = x1f if(y1f>y2f) else x2f
    distLine1 = xHalf - xLine1
    slopeM  =(y2f-y1f)/(x2f-x1f)
    xHalf = image.shape[1]/2
    yMax = image.shape[0]/2
    x1f2=-1
    y1f2=-1
    x2f2=-1
    y2f2=-1
    distThreshold = 0.2
    laneLength = 0
    laneDistFromCenter = math.inf
#    slopeM1 = math.inf
    lenLineSelect = 0
    if linesWhite is not None:
        for line in linesWhite:
            for x1,y1,x2,y2 in line:
                if ((x2-x1)  != 0):
                    m = (y2-y1)/(x2-x1)
                else:
                    m = math.inf
                c = y1-m*x1
                yi = yMax
                if m != 0 and m < math.inf:
                    xi = (yi - c)/m
                else:
                    continue
                laneLenLocal = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                dist = (xi - xHalf)
                slopeT1 = m
                slopeS = slopeM
                if(slopeM<0):
                    slopeS = -slopeS
                    slopeT1 = -m
                angleThreshold = 0.3
                angleMatch = 0
                distMach = 0
                if (abs(slopeS + slopeT1) <= abs(slopeS)*angleThreshold) and ((slopeS>0 and slopeT1<0) or (slopeS<0 and slopeT1>0)):
                    angleMatch = 1    
                if (( dist>0 and distLine1<0 ) or (dist<0 and distLine1>0)):
                    distMach = 1
                if distMach and angleMatch and lenLineSelect<laneLenLocal:
                    lenLineSelect = laneLenLocal
                    x1f2 = x1
                    x2f2 = x2
                    y1f2 = y1
                    y2f2 = y2
        
    slopeLine1  =(y2f-y1f)/(x2f-x1f)
    cl1 = y1f-slopeLine1*x1f
    slopeLine2  =(y2f2-y1f2)/(x2f2-x1f2)
    cl2 = y1f2-slopeLine2*x1f2
    
    l1y1 = int(image.shape[0])
    l2y1 = int(image.shape[0])
    
    l1x1 = (l1y1 - cl1)/slopeLine1
    l2x1 = (l2y1 - cl2)/slopeLine2
    
    xi = -1*(cl1-cl2) / (slopeLine1 - slopeLine2)
    yi = (cl2*slopeLine1 - cl1*slopeLine2) / (slopeLine1 - slopeLine2)
    
    xi = int(xi)
    yi = int(yi)
    l1x1 = int(l1x1)
    l2x1 = int(l2x1)
    
    
#    if x1>=0:
#        cv2.line(line_image,(x1f,y1f),(x2f,y2f),(255,255,0),10)
#    if x1f>=0:
#        cv2.line(line_image,(x1f2,y1f2),(x2f2,y2f2),(255,255,0),10)
    
    if x1f>=0:
        cv2.line(line_image,(l1x1,l1y1),(xi,yi),(255,0,0),10)

    
    if x1f2>=0:        
        cv2.line(line_image,(l2x1,l2y1),(xi,yi),(255,0,0),10)
        
    return line_image



strWorkDir = "test_images/"
strOpDir = "test_images_output/"


try:
    os.stat(strOpDir)
except:
    os.mkdir(strOpDir) 

arrInputImages = os.listdir("test_images/")

print(arrInputImages)
nCount = 0
for img in arrInputImages:
    print (strWorkDir +img)
    plt.figure(nCount)
    image = mpimg.imread(strWorkDir +img)
    line_img = extractMajorLaneMarking(image)
    #plt.set_title(img)
    #plt.subplot(1,4,1)
    #plt.imshow(image)
    #plt.subplot(1,4,2)
    combo = cv2.addWeighted(image, 0.8, line_img, 1, 0) 
    plt.imshow(combo)
    mpimg.imsave(strOpDir +img,combo)
    nCount += 1





