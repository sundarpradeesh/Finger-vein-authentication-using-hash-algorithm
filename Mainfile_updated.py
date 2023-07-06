# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy
import os

from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
from skimage.io import imshow


# === GETTING INPUT IMAGE


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filename1 = askopenfilename()

img = mpimg.imread(filename1)

# Image Resize
#row = 224 , column = 224

img_resize_ORIG = cv2.resize(img,((224, 224)))

plt.imshow(img)
plt.title('ORIGINAL IMAGE')
plt.show()

plt.imshow(img_resize_ORIG)
plt.title('RESIZED IMAGE')
plt.show()

# ======================================================
# Detect WLD , RL, MC
img = img_resize_ORIG
def first_filter(img):
    img_Blur=cv2.blur(img,(5,5))

    return img, img_Blur

# def edge_detection(img):
    #img = cv2.imread(file, 0)
    #img = cv2.imread("01.jpg", 0)
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
img_edge = cv2.addWeighted(absX,0.5,absY,0.5,0)
img_edge1 = cv2.addWeighted(absX,0.9,absY,0.9,0)

#cv2.imshow("absX", absX)
#cv2.imshow("absY", absY)
#cv2.imshow("Result", img_edge)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
fig = plt.figure(figsize = (30, 30))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
#ax3 = fig.add_subplot(1, 3, 3)
ax1.imshow(img, cmap = plt.cm.gray)
ax2.imshow(img_edge, cmap = plt.cm.gray)
plt.show()

    # return img, img_edge
# img_edge = edge_detection(img)

plt.imshow(img_edge)
plt.title('RLT')
plt.show()
plt.imshow(img_edge1)
plt.title('MC')

plt.show()
############################################
# def pixel_polarization(img_edge, img, threshold): # threshold 
img_edge1 = cv2.cvtColor(img_edge,cv2.COLOR_BGR2GRAY)

for i in range(len(img_edge1)):
    for j in range(len(img_edge1)):
        if img_edge1[i][j] > 100:
            img_edge1[i][j] = 255
        else:
            img_edge1[i][j] = 0

fig = plt.figure(figsize = (16, 16))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.imshow(img, cmap = plt.cm.gray)
ax2.imshow(img_edge1, cmap = plt.cm.gray)
plt.show()

ret, thresh1 = cv2.threshold(img_edge1, 1, 255, cv2.THRESH_BINARY)
plt.imshow(thresh1)
plt.title('WLD IMAGE')
plt.show()

img_edge_polar = img_edge
    # return img_edge_polar

def positioning_middle_point(img, dst, point_pixel):
    h, w = img.shape
    w1 = w // 5 
    w2 = (w // 5) * 4 
    '''
    print("roi width: ",h, w1, w2)
    '''
    low_l = False
    high_l = False
    while (not low_l or not high_l) and w1 < (w // 2):
        for i, pix in enumerate(dst[:, w1]):
            if i+1 < (h // 2) and not low_l:
                if pix == 255:
                    low_l = True
                    lower_left = i
            elif i+1 > (h // 2) and not high_l:
                h_h = int(h * (3/2) - (i+1)) 
                '''
                print(h_h)
                '''
                if dst[h_h, w1] == 255:
                    high_l = True
                    higher_left = h_h
        if not low_l or not high_l:
            w1 = w1 + 2
    middle_left = (lower_left + higher_left) // 2
    
    low_r = False
    high_r = False
    while (not low_r or not high_r) and w2 > (w // 2):
        for i, pix in enumerate(dst[:, w2]):
            if i+1 < (h // 2) and not low_r:
                if pix == 255:
                    low_r = True
                    lower_right = i
            elif i+1 > (h // 2) and not high_r:
                h_h = int(h * (3/2) - (i+1))
                if dst[h_h, w2] == 255:
                    high_r = True
                    higher_right = h_h
        if not low_r or not high_r:
            w2 = w2 - 2
    middle_right = (lower_right + higher_right) // 2
    
    dst[middle_left, w1] = point_pixel
    dst[middle_left+1, w1] = point_pixel
    dst[middle_left-1, w1] = point_pixel
    dst[middle_left, w1 + 1] = point_pixel
    dst[middle_left, w1 - 1] = point_pixel
    dst[middle_right, w2] = point_pixel
    dst[middle_right+1, w2] = point_pixel
    dst[middle_right-1, w2] = point_pixel
    dst[middle_right, w2 + 1] = point_pixel
    dst[middle_right, w2 - 1] = point_pixel
    
    fig = plt.figure(figsize = (16, 16))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img, cmap = plt.cm.gray)
    ax2.imshow(dst, cmap = plt.cm.gray)
    plt.show()
    
    return dst, middle_left, middle_right, w1, w2



#################################
def rotation_correction(img, dst, middle_right, middle_left, w1, w2):
    tangent_value = float(middle_right - middle_left) / float(w2 - w1)
    rotation_angle = np.arctan(tangent_value)/math.pi*180
    (h,w) = img.shape
    center = (w // 2,h // 2)
    M = cv2.getRotationMatrix2D(center,rotation_angle,1)
    rotated_dst = cv2.warpAffine(dst,M,(w,h))
    rotated_img = cv2.warpAffine(img,M,(w,h))
    '''
    fig = plt.figure(figsize = (16, 16))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(img, cmap = plt.cm.gray)
    ax2.imshow(rotated_dst, cmap = plt.cm.gray)
    ax3.imshow(rotated_img, cmap = plt.cm.gray)
    plt.show()
    '''
    return rotated_dst, rotated_img


def roi(rotated_img, rotated_edge, w1, w2, url):
    h, w = rotated_edge.shape
    r = range(0, h)
    r1 = range(0, h // 2)
    r2 = range(h // 2, h - 1)
    c = range(0, w)
    c1 = range(0, w // 2)
    c2 = range(w // 2, w-1)

    highest_edge = (rotated_edge[r1][:,c].sum(axis=1).argmax())
    lowest_edge = (rotated_edge[r2][:,c].sum(axis=1).argmax() + (h // 2))
    '''
    leftest_edge = (rotated_edge[r][:,c1].sum(axis=0).argmax())
    rightest_edge = (rotated_edge[r][:,c2].sum(axis=0).argmax() + (w // 2))
    '''
    leftest_edge = w1
    rightest_edge = w2
    '''
    _, img_w = rotated_edge.shape
    half = int(img_w/2)
    max_right_sum = 0
    max_right_i = 0
    sum_img = numpy.sum(rotated_img,axis=0)
    for i in range(half,img_w-50):
        s = sum(sum_img[i:i+50])
        if s > max_right_sum:
            max_right_sum = s
            max_right_i = i
    '''

    #print(highest_edge, lowest_edge, leftest_edge, rightest_edge)
    #print max_right_i
    #rightest_edge = max_right_i + 200
    #leftest_edge = 0 
    '''
    rotated_edge[highest_edge, : ] = 200
    rotated_edge[lowest_edge, : ] = 200 #150
    rotated_edge[: , leftest_edge] = 200 #200
    rotated_edge[: , rightest_edge] = 200 #250
    rotated_croped = rotated_edge[highest_edge : lowest_edge, leftest_edge : rightest_edge]
    '''
    rotated_croped_img = rotated_img[highest_edge : lowest_edge, leftest_edge : rightest_edge]
    '''
    fig = plt.figure(figsize = (30, 30))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.imshow(rotated_edge, cmap = plt.cm.gray)
    ax2.imshow(rotated_croped, cmap = plt.cm.gray)
    ax3.imshow(rotated_img, cmap = plt.cm.gray)
    ax4.imshow(rotated_croped_img, cmap = plt.cm.gray)
    plt.show()
    '''
    #print("rotated_croped_img type: ", rotated_croped_img)
    #cv2.imwrite(url, rotated_croped_img)

    #im = Image.fromarray(rotated_croped_img)
    #im.save(url)
    return rotated_croped_img
    
def img_resized_enhance(img, url):
    #resized_img = cv2.resize(img, (136, 100), cv2.INTER_NEAREST) #最近邻插值
    resized_img = cv2.resize(img, (320, 240), cv2.INTER_LINEAR) #双线性插值
    #resized_img = cv2.resize(img, (136, 100), cv2.INTER_NEAREST) #最近邻插值
    '''
    fig = plt.figure(figsize = (30, 20))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img, cmap = plt.cm.gray)
    ax2.imshow(resized_img, cmap = plt.cm.gray)
    plt.show()
    '''
    norm_resized_img = resized_img
    norm_resized_img = cv2.normalize(resized_img, norm_resized_img, 0, 255, cv2.NORM_MINMAX)
    #equ_resized_img = cv2.equalizeHist(resized_img)
    
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_resized_img = clahe.apply(norm_resized_img)
    '''
    plt.figure(figsize = (30, 30))
    plt.subplot(2, 2, 1), plt.title('image')
    plt.imshow(img, cmap = plt.cm.gray)
    plt.subplot(2, 2, 2), plt.title('resized_img')
    plt.imshow(resized_img, cmap = plt.cm.gray)
    plt.subplot(2, 2, 3), plt.title('norm_resized_img')
    plt.imshow(norm_resized_img, cmap = plt.cm.gray)
    plt.subplot(2, 2, 4), plt.title('CLAHE')
    plt.imshow(clahe_resized_img, cmap = plt.cm.gray)
    plt.show()
    '''
    print('saving...')
    cv2.imwrite(url, clahe_resized_img)
    print('done')
    return clahe_resized_img

def get_imgs_roi(img_file):
    images = os.listdir(img_file)
    for i, image in enumerate(images):
        print(i)
        print(image)
        img_raw = cv2.imread(os.path.join(img_file, image), 0)
        print(img_raw.shape)
        '''
        (h,w) = img.shape
        center = (w / 2,h / 2)
        M = cv2.getRotationMatrix2D(center,90,1)
        img_raw = cv2.warpAffine(img,M,(w,h))
        '''
        #img_raw, img_edge = edge_detection(img_raw)
        img_raw, img_Blur = first_filter(img_raw)
        img_raw, img_Blur_edge = edge_detection(img_Blur)
        
        fig = plt.figure(figsize = (50, 15))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.imshow(img_raw, cmap = plt.cm.gray)
        ax2.imshow(img_edge, cmap = plt.cm.gray)
        ax3.imshow(img_Blur_edge, cmap = plt.cm.gray)
        plt.show()
        
        img_Blur_edge_polar = pixel_polarization(img_Blur_edge, img_raw, 25) 
        img_Blur_edge_polar_midd, middle_left, middle_right, w1, w2= positioning_middle_point(img_raw, img_Blur_edge_polar, 100)
        img_Blur_edge_polar_midd_rotated, rotated_img = rotation_correction(img_raw, img_Blur_edge_polar_midd, middle_right, middle_left, w1, w2)
        new_file = './roi_600_2_all_320240'
        save_root = os.path.join(new_file,image)
        roi_img = roi(rotated_img, img_Blur_edge_polar_midd_rotated, w1, w2, save_root)
        resized_roi_img = img_resized_enhance(roi_img, save_root)


def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):
        params = {'ksize':(ksize, ksize), 'sigma':3.3, 'theta':theta, 'lambd':18.3,
                  'gamma':4.5, 'psi':0.89, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters
filters = build_filters()

def getGabor(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

accum = getGabor(img, filters)

################################################################
def bin_features_extract(roi_file):

    img_roi_raw = cv2.imread(roi_file, 0)
    filters = build_filters()
    img_roi_raw_gabor = getGabor(img_roi_raw, filters)
        #print(img_roi_raw_gabor)
        #norm_resized_img = cv2.normalize(img_roi_raw_gabor, norm_resized_img, 0, 255, cv2.NORM_MINMAX)
    #img_roi_raw_gabor_polar60 = img_roi_raw_gabor.copy()
    #img_roi_raw_gabor_polar60 = pixel_polarization(img_roi_raw_gabor_polar60, img_roi_raw, 60)
    img_roi_raw_gabor_polar70 = img_roi_raw_gabor.copy()
    img_roi_raw_gabor_polar70 = pixel_polarization(img_roi_raw_gabor_polar70, img_roi_raw, 70)

    return img_roi_raw_gabor_polar70
        
def bin_match(img1_path, img2_path):
    img1 = bin_features_extract(img1_path)
    img2 = bin_features_extract(img2_path)
    height, width = img1.shape
    size = height * width
    score = 0
    for i in range(len(img1)):
        for j in range(len(img1[i,:])):
            if img1[i][j] == img2[i][j]:
                score += 1
    scores = 100 * round((score / size), 4)
    #print(img1_path, img2_path, scores)
    return scores
        


def cut_image(image, m, n):
    height, width = image.shape
    item_width = int(width // m)
    item_height = int(height // n)
    #box_list = []
    cropped_list = []
    # (left, upper, right, lower)
    for i in range(0,n):
        for j in range(0,m):
            #print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            #box = (j*item_width,i*item_height,(j+1)*item_width,(i+1)*item_height)
            #box_list.append(box)
            cropped = image[i*item_height:(i+1)*item_height, j*item_width:(j+1)*item_width]
            cropped_list.append(cropped)
            
    print(len(cropped_list))
    #image_list = [image.crop(box) for box in box_list]
    return cropped_list


from skimage.feature import local_binary_pattern


##########################################
def SIFT_detector(gray_path):
    images_sift = os.listdir(gray_path)
    for i, image_sift in enumerate(images_sift):
        print(i)
        print(image_sift)
        img = cv2.imread(os.path.join(gray_path, image_sift), 0)

        
        kaze = cv2.KAZE_create()
        kp = kaze.detect(img,None)
        img_kaze=cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        #cv2.imwrite('sift_keypoints.jpg',img)
        
        plt.figure(figsize = (30, 30))
        plt.subplot(1, 2, 1), plt.title('img')
        plt.imshow(img, cmap = plt.cm.gray)
        plt.subplot(1, 2, 2), plt.title('img_kaze')
        plt.imshow(img_kaze, cmap = plt.cm.gray)
#        plt.subplot(1, 3, 3), plt.title('lbp_hist')
#        plt.imshow(lbp_hist)
        plt.show()
    
def SIFT_match(img1_path, img2_path):
    
    img1 = cv2.imread(img1_path,0)          # queryImage
    img2 = cv2.imread(img2_path,0) # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

def FLANN_based_Matcher(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 0)          # queryImage
    img2 = cv2.imread(img2_path, 0) # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    #matchesMask = []
    # ratio test as per Lowe's paper
    match_keypoints_count = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            matchesMask[i]=[1,0]
            #matchesMask.append(m)
            match_keypoints_count += 1
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)

    score = 100 * round(match_keypoints_count / len(matchesMask), 4)
    #print('score = ', score)
    '''
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
    plt.imshow(img3),plt.show()
    '''
    return score

def cal_scores(method='FLANN', flag=1):
    scores_list_diff = []
    scores_list_same = []
    for k in range(1,5):
        if k is not flag:
            for i in range(1,11):
                for j in range(1,11):
                    #print('%s', )
                    strs1 = './data/roi_600_2_all_320240/600-{}-{}-1.bmp'.format(flag,i)
                    strs2 = './data/roi_600_2_all_320240/600-{}-{}-1.bmp'.format(k,j)
                    if method == 'FLANN':
                        scores = FLANN_based_Matcher(strs1, strs2)
                        scores_list_diff.append(scores)
                    if method == 'BIN':
                        scores = bin_match(strs1, strs2)
                        scores_list_diff.append(scores)
                    print(strs1,strs2, scores)
                        
            
    for i in range(1,11):
        for j in range(1,11):
            #print('%s', )
            strs1 = './data/roi_600_2_all_320240/600-{}-{}-1.bmp'.format(flag,i)
            strs2 = './data/roi_600_2_all_320240/600-{}-{}-1.bmp'.format(flag,j)
            
            if method == 'FLANN':
                scores = FLANN_based_Matcher(strs1, strs2)
                scores_list_same.append(scores)
            if method == 'BIN':
                scores = bin_match(strs1, strs2)
                scores_list_same.append(scores)
            print(strs1,strs2, scores)
            
    plt.hist(scores_list_diff, 60, range=(0,100), density=True, histtype="bar", facecolor='g', label='Inter-class', alpha=0.5)
    plt.hist(scores_list_same, 60, range=(0,100), density=True, histtype="bar", facecolor='r', label='In-class', alpha=0.5)
    plt.xlabel('Matched Features Ratio(MFR)(%)', fontsize=25)
    plt.ylabel('MFR Histogram', fontsize=25)
    plt.title('Distribution of matching ratio between in-class samples and inter-class samples', fontsize=30)
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([0, 1, 0, 0.03])
    plt.grid(True)
    plt.show()

#print(scores_list)
        
# get_imgs_roi('./Dataset/Finger Vein/')
#bin_features_extract('./data/roi_320240')
#bin_match('./data/roi_600_2_all_320240/600-3-7-1.bmp', './data/roi_600_2_all_320240/600-3-8-1.bmp')
#LBP_feature_extrector('./data/roi_320240')
#SIFT_detector('./data/roi_320240/')
#SIFT_match('./data/roi_320240/600-3-7-1.bmp', './data/roi_320240/600-3-8-1.bmp')
#cal_scores('FLANN', 2)
#cal_scores('BIN', 4)

import cv2
import numpy as np
from matplotlib import pyplot as plt

	
def get_pixel(img, center, x, y):
	
	new_value = 0
	
	try:
		# If local neighbourhood pixel
		# value is greater than or equal
		# to center pixel values then
		# set it to 1
		if img[x][y] >= center:
			new_value = 1
			
	except:
		# Exception is required when
		# neighbourhood value of a center
		# pixel value is null i.e. values
		# present at boundaries.
		pass
	
	return new_value

# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):

	center = img[x][y]

	val_ar = []
	
	# top_left
	val_ar.append(get_pixel(img, center, x-1, y-1))
	
	# top
	val_ar.append(get_pixel(img, center, x-1, y))
	
	# top_right
	val_ar.append(get_pixel(img, center, x-1, y + 1))
	
	# right
	val_ar.append(get_pixel(img, center, x, y + 1))
	
	# bottom_right
	val_ar.append(get_pixel(img, center, x + 1, y + 1))
	
	# bottom
	val_ar.append(get_pixel(img, center, x + 1, y))
	
	# bottom_left
	val_ar.append(get_pixel(img, center, x + 1, y-1))
	
	# left
	val_ar.append(get_pixel(img, center, x, y-1))
	
	# Now, we need to convert binary
	# values to decimal
	power_val = [1, 2, 4, 8, 16, 32, 64, 128]

	val = 0
	
	for i in range(len(val_ar)):
		val += val_ar[i] * power_val[i]
		
	return val

#  ================================================
#  Pattern Extraction

img_bgr = img
height, width, _ = img_bgr.shape

# We need to convert RGB image
# into gray one because gray
# image has one channel only.
img_gray = cv2.cvtColor(img_bgr,
						cv2.COLOR_BGR2GRAY)

# Create a numpy array as
# the same height and width
# of RGB image
img_lbp = np.zeros((height, width),
				np.uint8)

for i in range(0, height):
	for j in range(0, width):
		img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

plt.imshow(img_bgr)
plt.show()

plt.imshow(img_lbp, cmap ="gray")
plt.title('Vein Patterns')
plt.show()

#  ================================
# Train Test Splitting

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


test_data1 = os.listdir('Dataset/Finger Vein/Yes/')
test_data2 = os.listdir('Dataset/Finger Vein/No/')

dot= []
labels_target = []

for img in test_data1:
    
    try:
        img_1 = mpimg.imread('Dataset/Finger Vein/Yes' + "/" + img)
        img_resize = cv2.resize(img_1,((224, 224)))
        dot.append(np.array(img_resize))
        labels_target.append(0)
        
    except:
        None
        
for img in test_data2:
    
    try:
        img_2 = mpimg.imread('Dataset/Finger Vein/No/'+ "/" + img)
        img_resize = cv2.resize(img_2,(224, 224))
        
        dot.append(np.array(img_resize))
        labels_target.append(1)
        
    except:
        None
        
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dot,labels_target,test_size = 0.2, random_state = 101)

x_train1=np.zeros((len(x_train),224,224,3))

try:
    
    for i in range(0,len(x_train)):
            x_train1[i,:,:,:]=x_train[i]
except:
        
            x_train1[i,:,:]=x_train[i]

x_test1=np.zeros((len(x_test),224,224,3))

try:
        
    for i in range(0,len(x_test)):
            x_test1[i,:,:,:]=x_test[i]     
except:

            x_test1[i,:,:]=x_test[i]
            
# =========================================
# Deep Learning
# ======== CNN ===========
    
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
# from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dropout
from keras.utils import to_categorical




# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)


print("-------------------------------------")
print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("-------------------------------------")
print()
#fit the model 
history=model.fit(x_train1,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)

accuracy = model.evaluate(x_train1, train_Y_one_hot, verbose=1)

pred_cnn = model.predict([x_train1])

y_pred2 = pred_cnn.reshape(-1)
y_pred2[y_pred2<0.5] = 0
y_pred2[y_pred2>=0.5] = 1
y_pred2 = y_pred2.astype('int')

# loss=history.history['loss']
loss = accuracy[0]
# loss=max(loss)

acc_cnn=100-loss

print("-------------------------------------")
print("PERFORMANCE ---------> (CNN)")
print("-------------------------------------")
print()
#acc_cnn=accuracy[1]*100
print("1. Accuracy   =", acc_cnn,'%')
print()
print("2. Error Rate =",loss)


#  =========================
#  Auto Encoder
# =========================
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
import os

import numpy as np


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dot,labels_target,test_size = 0.2, random_state = 101)


x_train2=np.zeros((len(x_train),224,224))

try:
    
    for i in range(0,len(x_train)):
            tempt = x_train[i]
            x_train2[i,:,:,:]=tempt[:,:,0]
except:
            tempt = x_train[i]
        
            x_train2[i,:,:]=tempt[:,:,0]

x_test2=np.zeros((len(x_test),224,224))

try:
        
    for i in range(0,len(x_test)):
            tempt = x_test[i]
        
            x_test2[i,:,:,:]=tempt[:,:,0]     
except:
            tempt = x_test[i]

            x_test2[i,:,:]=tempt[:,:,0]
            
            
x_train = x_train2.astype('float32') / 255
x_test = x_test2.astype('float32') / 255
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print(x_train.shape)
print(x_test.shape)

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)



import matplotlib.pyplot as plt
# %matplotlib inline
n = 10
plt.figure(figsize=(20, 2))
for i in range(1,n+1):
    ax = plt.subplot(1, n, i)
    temp_ns = cv2.resize(x_test_noisy[i],(28, 28))
    plt.imshow(temp_ns)
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    plt.show()


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
input_img = Input(shape=(224, 224,1))  # adapt this if using `channels_first` image data format
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
# at this point the representation is (7, 7, 32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


autoencoder.fit(x_train_noisy, x_train,
                epochs=1,
                batch_size=224,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = autoencoder.predict(x_test)
decoded_imgs = autoencoder.predict(encoded_imgs)
encoded_imgs[0].shape

# ========== Encoded image
plt.imshow(encoded_imgs[0])
plt.title('Encoded Image')
plt.show()
decoded_imgs[0].shape

# ============ Decoded Image
plt.imshow(decoded_imgs[0])
plt.title('Decoded Image')
plt.show()


#  ===========================
# Hashing Step
import random

import base64 
# image = dot[0] #open binary file in read mode
# step 1
template = decoded_imgs
# step 2
np.shape(template)
# step 3
np.shape(encoded_imgs)
#  step4
user_seed = random.randint(3, 9)
# step 8,9,10,11

import numpy as np

def gramschmidt(A):
    """
    Applies the Gram-Schmidt method to A
    and returns Q and R, so Q*R = A.
    """
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j]*Q[:, k]
    return Q, R

def main():
    """
    Prompts for n and generates a random matrix.
    """
    global Q
    A = template[:,:,0]
    print('A = ')
    print(A)
    Q, R = gramschmidt(temp_ns)
    print(np.dot(Q.transpose(), Q))
    print('Q*R Transpose =')
    print(np.dot(Q, R))
    
main()


image_64_encode = base64.b64encode(Q)
# print(image_64_encode)

# ========================
# Machine Learning
if image_64_encode !=[]:
        
    from sklearn import svm
    clf = svm.SVC() # apply SVM
    
    
    All_features = []
    for ii in range(0,len(dot)):
        MN = np.mean(dot[ii])
        ST = np.std(dot[ii])
        VR = np.var(dot[ii])
        arr = [MN,ST,VR]
        All_features.append(arr)
        
    clf.fit(All_features, labels_target)
    MN = np.mean(img_resize_ORIG)
    ST = np.std(img_resize_ORIG)
    VR = np.var(img_resize_ORIG)
    Testfea = [MN,ST,VR]
        
    Class_val = clf.predict([Testfea])
        
    # temp = np.mean(dot[ii] - img_resize_ORIG)
    # if temp == 0:
    #     Class = ii
            
    if labels_target[Class_val[0]] == 0:
        print('---------------')
        print('Authenticated')
        print('---------------')
    
    elif labels_target[Class_val[0]] == 1:
        print('---------------')
        print('Not Authenticated')
        print('---------------')


import matplotlib.pyplot as plt

Class_val1 = clf.predict(All_features)


plt.scatter(np.arange(0,len(Class_val1)),Class_val1)
plt.title('Cluster')
plt.grid()
plt.show()
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
plot_roc_curve(Class_val1, labels_target)
plt.title('ROC')
plt.grid()
plt.show()


plt.plot(np.arange(0,len(x_test2)))
plt.title('Cluster')
plt.grid()
plt.show()

plt.plot(Testfea)
plt.title('Cluster')
plt.grid()
plt.show()

# plot_roc_curve(Testfea, Testfea)
# plt.title('ROC')
# plt.grid()
# plt.show()

reted,threshed = cv2.threshold(img_edge,1,0,2)

plot_roc_curve(np.array(threshed[:,:,0][1]), np.array(threshed[:,:,0][1]))
plt.title('RLT ROC')
plt.grid()
plt.show()


reted,threshed = cv2.threshold(img_edge1,1,255,2)

plot_roc_curve(np.array(threshed[32]), np.array(threshed[32]))
plt.title('RLT ROC')
plt.grid()
plt.show()


reted,threshed = cv2.threshold(thresh1,1,255,2)

plot_roc_curve(np.array(threshed[32]), np.array(threshed[33]))
plt.title('WLT ROC')
plt.grid()
plt.show()



