import cv2
import matplotlib.pyplot as plt 
import time
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from scipy import spatial
import fitz
import os
import glob, imutils
%matplotlib inline

os.chdir('/home/vishal/hist')

## Reading Images from PDFs

doc = fitz.open("5.pdf")
for i in range(len(doc)):
    for img in doc.getPageImageList(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n < 5:       # this is GRAY or RGB
            pix.writePNG("p%s.png" %i)
        else:               # CMYK: convert to RGB first
            pix1 = fitz.Pixmap(fitz.csRGB, pix)
            pix1.writePNG("p%s.png" %i)
            pix1 = None
        pix = None

### Skew Correction via rotation ###
  
def rotate(image, angle):
    print image.shape
    if angle>0:
        return(imutils.rotate_bound(image, (360-angle)))
    else: 
        return(imutils.rotate_bound(image, (-angle)))

### Horizontal Histogram Projection ###
def hproject_hist(bw):
    counts = np.sum(bw==0, axis=1)
    row_number = [i for i in range(bw.shape[1])]
    return counts

### Vertical Histogram Projection ###
def vproject_hist(bw):
    counts = np.sum(bw==0, axis=0)
    col_number = [i for i in range(bw.shape[0])]
    return counts


### Pattern Tracking ###
def vert_roi(x):                                  #Region of Interest#
    length = len(x)
    vroi = []
    boundaries = []
    for i in range(length):
        if (x[i]!=min(x)):
            vroi.append(i)
    boundaries = [vroi[0]]
    for x in range(len(vroi)-1):
        if (vroi[x+1] != (vroi[x] + 1)):
            boundaries.append(vroi[x])
            boundaries.append(vroi[x+1])
    boundaries.append(vroi[len(vroi)-1])
    region = []
    percentage = []
    for p in range(len(boundaries)/2):
        z = [boundaries[2*p], boundaries[(2*p)+1]]
        region.append(z)
        y = round((boundaries[2*p]/float(length))*100, 3)
        percentage.append(y)
    return vroi, region, percentage


def signalgeneration(image, boundary):
    signals = []
    for z in range(len(boundary)):
        up, down = boundary[z]
#         print down
        region = image[up:down, 0:499]
#         cv2.imshow("cropped",region)
#         cv2.waitKey(0)
        signal = vproject_hist(region)
        signals.append(signal)
    return signals

### Criteria 1 - kMeans ###
def label(signals):
    z = np.zeros(shape = (len(signals), len(signals)))
#         y = np.zeros(shape = (len(signals), 1))
    for n in range(len(signals)):
###            Using Cosine Similarity
        for m in range(len(signals)):
            cos_dis = round(1 - spatial.distance.cosine(signals[m], signals[n]), 3)
            z[n][m] = cos_dis
    
    z = np.nan_to_num(z)
    mean_cosine = []
    sd_cosine = []
    for r in range(len(signals)):
        mean_cosine.append(round(np.mean(z[r]),3))
        sd_cosine.append(round(np.std(z[r]),3))
            
    plot = []
    nlines = len(mean_cosine)
    for v in range(nlines):
        a = [mean_cosine[v], sd_cosine[v]]
        plot.append(a)
    plt.plot(mean_cosine, "r+")
    plt.show()
    kmeans = KMeans(n_clusters=2).fit(plot)
    print kmeans.labels_
    return kmeans.labels_

### Criteria 2 - Sucssesive Labelling ###
def sim_criteria(f,s):
    bv = round(1 - spatial.distance.cosine(f, s), 3)
    if (bv>.3):
        return 1
    else:
        return 0
    
def KMeans_neighbour(X):
    labels = [0]*len(X)
    label = 0
    for i,vals in enumerate(X):
        if i == 0:
            continue
        if not sim_criteria(X[i-1], X[i]):
            label += 1
        labels[i] = label
    print labels
    return labels


### Header Footer Removal ###
def remHeadFoot(labels, bound, image):
#     plt.imshow(image)
    
    r = 500.0 / image.shape[1]
    dim = (500, int(image.shape[0] * r))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    idxH = []
    idxF = []
## For Header
    for x in range(5):
        if (labels[x] != labels[x+1]):
            idxH.append(x+1)
## For Footer
    for y in range(5):
        if (labels[len(labels)-1-x] != labels[len(labels) - 2 - x]):
            idxF.append(len(labels)-2-x)
    
    im_info = image.shape
    print im_info
#     crop_img = image[61:250,0:499]
    if (len(idxF)==0):
        Hindex = min(idxH)
        print Hindex
        Hup, Hdown = bound[Hindex]
        print Hup
        crop_img = image[Hup:, : ]
    elif (len(idxH)==0 | len(idxF)==0):
        crop_img = image 
    else:
        Hindex = min(idxH)
        Findex = min(idxF)
        Hup, Hdown = bound[Hindex]
        Vup, Vdown = bound[Findex]
        crop_img = image[Hup:Vdown, : ]
    return crop_img


### Main Operation ###
fpaths = glob.glob("/home/vishal/hist/*.png")
processed_Images=[]
### Page-wise Operation ###
for fpath in fpaths:
    img = cv2.imread(fpath,0)
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50, minLineLength=50, maxLineGap=10)
    
    ## Skew Correction
    
    x = []
    y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if ((x2-x1) == 0):
                break
            else:
                z = (y2-y1)/float(x2-x1)
                if (-.175<(z)<.175):
                #cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                    y.append(y2-y1)
                    x.append(x2-x1)
    #x1,y1,x2,y2 = line[0]
    x0 = sum(x)/float(len(x))
    y0 = sum(y)/float(len(y))
    slope = y0/x0
    angle = ((np.arctan(slope)*180)/np.pi)
    print 'angle',angle
    rotated_im = rotate(img, angle)                         #Rotation
    print 'rotated_im.shape',rotated_im.shape
    height, width = rotated_im.shape
    data = rotated_im[10:(height-10), 10:(width-10)]           #Resize the image
    print 'data.shape',data.shape
    
    ## Resizing the image for projection
    
    r = 500.0 / float(data.shape[1])
    dim = (500, int(data.shape[0] * r))
    image = cv2.resize(data, dim, interpolation = cv2.INTER_NEAREST)
    plt.imshow(image)
    
    ## Projection Method
    
    thresh, bwimage = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)            # Binary Conversion
    
    hdata = hproject_hist(bwimage)                                                 # Horizontal Projection
    roi, bound, percentage = vert_roi(hdata)
    print bound
    signals = signalgeneration(bwimage, bound)                                     # Projection and Cutting
    labelling = label(signals)
    proIm = remHeadFoot(labelling, bound, image)
    processed_Images.append(proIm)
    i = i+1
