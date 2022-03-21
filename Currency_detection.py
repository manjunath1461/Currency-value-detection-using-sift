import cv2
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
max_val = 8
max_pt = 0
max_kp = 0

sift = cv2.SIFT_create()# initializin SIFT object

#input is given as a image


#test_img = cv2.imread('Test/10/3.jpg') #using opencv to read the image file 
#test_img = cv2.imread('Test/20/2.jpg')
#test_img = cv2.imread('Test/50/3.jpg')
#test_img = cv2.imread('Test/100/1.jpg')
#test_img = cv2.imread('Test/200/3.jpg')
#test_img = cv2.imread('Test/500/3.jpg')
test_img = cv2.imread('Test/2000/3.jpg')
cv2.imshow('test image',test_img)

# resizing must be dynamic
original =cv2.resize(test_img, None, fx=0.4, fy=0.4, interpolation = cv2.INTER_AREA)# resizing the image
cv2.imshow('original.jpg',original)
cv2.imwrite('original.jpg',original)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) #converting image to gray-scale 
cv2.imshow('img_gray.jpg',img_gray)
cv2.imwrite('img_gray.jpg',img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0) # image smoothing using gaussian blur
cv2.imshow('img_gblur.jpg',img_blur)
cv2.imwrite('img_gblur.jpg',img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


# keypoints and descriptors
(kp1, des1) = sift.detectAndCompute(img_blur,None) #calculating key-point and the descriptors


# training the images which are in train folder for helping in detecting the value of currency

training_set = ['Train/500_16.jpg','Train/2000_19.jpg','Train/200_8.jpg','Train/50_13.jpg','Train/100_10.jpg','Train/2000_25.jpg','Train/20_15.jpg','Train/50_16.jpg','Train/50_15.jpg','Train/100_28.jpg','Train/100_4.jpg','Train/20_2.jpg','Train/100_13.jpg','Train/500_7.jpg','Train/50_12.jpg','Train/100_23.jpg','Train/2000_12.jpg','Train/100_3.jpg','Train/100_22.jpg','Train/100_5.jpg','Train/2000_2.jpg','Train/10_20.jpg','Train/200_6.jpg','Train/100_6.jpg','Train/2000_4.jpg','Train/200_25.jpg','Train/10_10.jpg','Train/10_3.jpg','Train/2000_27.jpg','Train/500_21.jpg','Train/10_26.jpg','Train/50_22.jpg','Train/20_12.jpg','Train/10_30.jpg','Train/200_1.jpg','Train/20_20.jpg','Train/500_19.jpg','Train/50_19.jpg','Train/2000_6.jpg','Train/100_29.jpg','Train/20_17.jpg','Train/200_12.jpg','Train/50_31.jpg','Train/200_18.jpg','Train/10_18.jpg','Train/10_6.jpg','Train/500_3.jpg','Train/500_25.jpg','Train/50_7.jpg','Train/20_10.jpg','Train/10_14.jpg','Train/20_1.jpg','Train/100_21.jpg','Train/50_11.jpg','Train/10_17.jpg','Train/200_5.jpg','Train/500_24.jpg','Train/200_27.jpg','Train/500_9.jpg','Train/20_6.jpg','Train/500_2.jpg','Train/200_13.jpg','Train/100_18.jpg','Train/500_20.jpg','Train/2000_8.jpg','Train/20_4.jpg','Train/10_16.jpg','Train/20_23.jpg','Train/50_26.jpg','Train/2000_1.jpg','Train/200_26.jpg','Train/2000_11.jpg','Train/500_5.jpg','Train/2000_22.jpg','Train/20_29.jpg','Train/2000_16.jpg','Train/10_27.jpg','Train/50_20.jpg','Train/200_21.jpg','Train/100_2.jpg','Train/200_28.jpg','Train/2000_10.jpg','Train/500_12.jpg','Train/10_25.jpg','Train/20_16.jpg','Train/50_21.jpg','Train/50_18.jpg','Train/20_8.jpg','Train/500_4.jgp','Train/2000_7.jpg','Train/50_1.jpg','Train/500_23.jpg','Train/100_20.jpg','Train/200_10.jpg','Train/100_25.jpg','Train/2000_26.jpg','Train/50_17.jpg','Train/10_5.jpg','Train/500_26.jpg','Train/20_3.jpg','Train/20_13.jpg','Train/20_28.jpg','Train/2000_29.jpg','Train/50_25.jpg','Train/20_9.jpg','Train/200_19.jpg','Train/100_24.jpg','Train/200_11.jpg','Train/200_20.jpg','Train/50_23.jpg','Train/50_2.jpg','Train/20_33.jpg','Train/500_18.jpg','Train/50_3.jpg','Train/500_8.jpg','Train/200_17.jpg','Train/20_11.jpg','Train/2000_13.jpg','Train/10_4.jpg','Train/200_7.jpg','Train/2000_20.jpg','Train/20_7.jpg','Train/100_27.jpg','Train/100_1.jpg','Train/10_24.jpg','Train/10_21.jpg','Train/200_2.jpg','Train/20_26.jpg','Train/2000_21.jpg','Train/10_12.jpg','Train/200_4.jpg','Train/10_15.jpg','Train/100_30.jpg','Train/200_30.jpg','Train/50_14.jpg','Train/2000_24.jpg','Train/500_29.jpg','Train/10_13.jpg','Train/500_28.jpg','Train/2000_17.jpg','Train/200_29.jpg','Train/500_15.jpg','Train/10_1.jpg','Train/100_7.jpg','Train/10_7.jpg','Train/500_22.jpg','Train/20_32.jpg','Train/100_11.jpg','Train/100_8.jpg','Train/500_13.jpg','Train/10_19.jpg','Train/50_6.jpg','Train/50_8.jpg']
start = time.time()
for i in range(0, len(training_set)):
	# train image
	train_img0 = cv2.imread(training_set[i])
	train_img =cv2.resize(train_img0, None, fx=0.4, fy=0.4, interpolation = cv2.INTER_AREA)# resizing the image
	img_gray2 = cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY)# grayscale conversion
	
	img_blur2 = cv2.GaussianBlur(img_gray2, (5, 5), 0) # gaussian blur

	(kp2, des2) = sift.detectAndCompute(img_blur2, None) # calculating keypoints and descriptors 

	# brute force matcher with default params
	
	bf = cv2.BFMatcher() # creating a BF matcher object
	all_matches = bf.knnMatch(des1, des2, k=2) #returns the k best matches
	
	
	# Apply ratio test
	good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.789 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if max_val != 8:
	print(training_set[max_pt])
	print('good matches ', max_val)

	train_img7 = cv2.imread(training_set[max_pt])
	cv2.imwrite('training_set.jpg',train_img7)
	train_img1 =cv2.resize(test_img, None, fx=0.4, fy=0.4, interpolation = cv2.INTER_AREA)# resizing the image
	img_gray3 = cv2.cvtColor(train_img1, cv2.COLOR_RGB2GRAY)# grayscale conversion
	
	img_blur3 = cv2.GaussianBlur(img_gray3, (5, 5), 0) # gaussian blur
	img3 = cv2.drawMatchesKnn(img_blur, kp1,img_blur3, max_kp, good, 4) #drawing of match lines for each key point and its best match points
	plt.imshow(img3),plt.show()
	note = str(training_set[max_pt])[6:-6]
	print('\nDetected denomination: Rs. ', note)

else:
	print('No Matches')
end = time.time()
print("Execution time:",end-start)
