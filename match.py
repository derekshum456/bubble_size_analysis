# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:28:59 2020


"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Image_10.jpg',0)
img2 = img.copy()
##template = cv2.imread('template102.png',0)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r = 155
thk = 8

tsz = 2*r + 2 * thk

template = np.zeros((tsz,tsz,1), np.uint8)

cv2.circle(template,(r+thk,r+thk),r,255,thk)
th3 = cv2.threshold(img,22,255,cv2.THRESH_BINARY)[1]

w = tsz
h = tsz


methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(th3,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(th3,top_left, bottom_right, 255, 2)
    
    
    res = cv2.threshold(res,22,255,cv2.THRESH_BINARY)[1]

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(th3,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()

"""

res = cv2.matchTemplate(img,tmpImage,cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(res,cmap = 'gray')
plt.title('Match Image'), plt.xticks([]), plt.yticks([])



# Apply template Matching
#res = cv2.matchTemplate(img,tmpImage,cv2.TM_CCOEFF_NORMED)
#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


#plt.subplot(221),plt.imshow(res,cmap = 'gray')
#plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(img,cmap = 'gray')
#plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

"""
plt.show()
