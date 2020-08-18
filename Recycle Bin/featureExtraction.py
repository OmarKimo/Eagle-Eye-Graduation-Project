import tifffile as tiff
import cv2
import numpy as np

def computeNormal2D(mat): #input must be a 2D array
    amin = np.amin(mat)
    mat = (mat - amin) / (np.amax(mat) - amin)
    return mat

def computeNormal3D(mat): #input must be a 3D array
    for i in range(mat.shape[2]):
        mat[:,:,i] = computeNormal2D(mat[:,:,i])
    return mat

def computeSMoment(mat,mean=None): #input must be a 2D array
  if mean == None:
    mean = mat.mean()
  return np.sqrt(mean)/np.square(mat).mean()

def convertRGB2HSV(img):
    img = img/1023.0
    imgHSV = img
    for i in range(len(img)):
        for j in range(len(img[0])):
            cmax = max(img[i][j][0],img[i][j][1],img[i][j][2])
            cmin = min(img[i][j][0],img[i][j][1],img[i][j][2])
            delta = cmax-cmin

            if cmax==cmin:
                imgHSV[i][j][0] = 0
            elif cmax == img[i][j][0]:
                imgHSV[i][j][0] = (60 * ((img[i][j][1] - img[i][j][2]) / delta) + 360) % 360
            elif cmax == img[i][j][1]:
                imgHSV[i][j][0] = (60 * ((img[i][j][2] - img[i][j][0]) / delta) + 120) % 360
            elif cmax == img[i][j][2]:
                imgHSV[i][j][0] = (60 * ((img[i][j][0] - img[i][j][1]) / delta) + 240) % 360
      
            if cmax==0:
                imgHSV[i][j][1]=0
            else:
                imgHSV[i][j][1]=(delta/cmax)
      
            imgHSV[i][j][2]=cmax

    return imgHSV

def computeCCMFeatures(img,band,windowSize = 7):

    if windowSize % 2 == 0:
        return
    paramCount = 0
    if band == 'H':
        paramCount = 3
        img = img*255/360
    elif band == 'S':
        paramCount = 2
        img = img*255
    elif band == 'V':
        paramCount = 2
        img = img*255
    else:
        return
  
    img = np.around(img)  
    img = img.astype(np.uint8)
  
    newImg = np.zeros((478,478,paramCount))
    halfWindow=windowSize//2
    ccm = np.zeros((256,256), dtype=np.uint8) #assuming windowSize <= 15
    a = 0
    b = 0
    for i in range(halfWindow,img.shape[0]-halfWindow,windowSize):
        for j in range(halfWindow,img.shape[1]-halfWindow,windowSize):
            # i,j = pixel in whole-image loop
            ccm = np.zeros((256,256), dtype=np.uint8) #assuming windowSize <= 15
            for k in range(i-halfWindow,min(img.shape[0]-1,i+halfWindow+1)):
                for l in range(j-halfWindow,min(img.shape[1]-1,j+halfWindow+1)):
                    # k,l = pixel in sliding window loop
                    ccm[img[k,l],img[k+1,l+1]] += 1
            mean = ccm.mean()
            newImg[a][b][0] = mean
            ccmSquared = np.square(ccm)
        
            if band == 'H':
                #sosvh is SSD
                newImg[a][b][1]=np.sum(np.square(ccm-mean)) #sosvh
                newImg[a][b][2]=np.sum(ccmSquared) #autoc
            else:
                newImg[a][b][1]=(np.sqrt(mean))/(ccmSquared.mean()) #smoment
                #if band == 'I':
                #  covariance = np.cov(ccm)[0][-1] ###### Check if it works for a matrix and replace with my function
                #  newImg[a][b][2]=covariance
            b = (b+1)%478
        a += 1
    return newImg


def computeHSVFeatures(img,windowSize = 7):
    paramCount = 7
    newImg = np.zeros((478,478,paramCount))
    halfWindow=windowSize//2
    a = 0
    b = 0
    for i in range(halfWindow,img.shape[0]-halfWindow,7):
        for j in range(halfWindow,img.shape[1]-halfWindow,7):
            # i,j = pixel in whole-image loop      
        
            minX=i-halfWindow
            maxX=i+halfWindow+1
            minY=j-halfWindow
            maxY=j+halfWindow+1

            newImg[a][b][4] = img[minX:maxX,minY:maxY,0].mean() #meanH
            newImg[a][b][6] = img[minX:maxX,minY:maxY,1].mean() #meanS
            newImg[a][b][5] = img[minX:maxX,minY:maxY,2].mean() #meanI
            newImg[a][b][0] = computeSMoment(img[minX:maxX,minY:maxY,2], newImg[a][b][5]) #smomentI
            newImg[a][b][1] = np.var(img[minX:maxX,minY:maxY,2]) #varianceI
            newImg[a][b][2] = np.sqrt(newImg[a][b][1]) #stdI
            newImg[a][b][3] = np.std(img[minX:maxX,minY:maxY,0]) #stdH
            b = (b+1)%478
        a += 1
        
    return newImg

def computeNIRFeatures(img,windowSize = 7):
    paramCount = 2
    newImg = np.zeros((478,478,paramCount))
    halfWindow=windowSize//2
    a = 0
    b = 0
    for i in range(halfWindow,img.shape[0]-halfWindow,7):
        for j in range(halfWindow,img.shape[1]-halfWindow,7):
            # i,j = pixel in whole-image loop      
            
            minX=i-halfWindow
            maxX=i+halfWindow+1
            minY=j-halfWindow
            maxY=j+halfWindow+1
            
            newImg[a][b][1] = img[minX:maxX,minY:maxY].mean() #meanNIR
            newImg[a][b][0] = np.sqrt(np.var(img[minX:maxX,minY:maxY])) #stdNIR
            b = (b+1)%478
        a += 1
    
    return newImg

def extractImageFeatures(imagePath):
    imgRGBN = np.zeros((3346, 3346, 4), "float32")
    # for type M 
    imgRGBN[..., 3] = cv2.resize(np.transpose(tiff.imread("{}_M.tif".format(imagePath[:-4])), (1,2,0))[:,:,7], (3346, 3346))
    #img_NIR = cv2.resize(img_NIR, (3346, 3346)) #stretch to fit 7*7

    # for RGB
    imgRGBN[..., 0:3] = cv2.resize(np.moveaxis(tiff.imread(imagePath),0,-1), (3346, 3346)) #compress to fit 7*7
    #img_RGB = img_RGB[:3347,:3347,:] 
    
    #print(f'RGB shape: {img_RGB.shape}\nM shape: {img_M.shape}\nimg shape: {img.shape}')
    
    #imgRGBN = stretch_n(imgRGBN)
    imgHSV = convertRGB2HSV(imgRGBN[:,:,0:3])
    img = np.zeros((478,478,16)) 
    
    temp = computeCCMFeatures(imgHSV[:,:,0],'H')
    #print("ccmH")
    img[:,:,4]=temp[:,:,0]
    img[:,:,1]=temp[:,:,1]
    img[:,:,2]=temp[:,:,2]

    temp = computeCCMFeatures(imgHSV[:,:,1],'S')
    #print("ccmS")
    img[:,:,3]=temp[:,:,0]
    img[:,:,5]=temp[:,:,1]

    temp = computeCCMFeatures(imgHSV[:,:,2],'V')
    #print("ccmI")
    img[:,:,0]=temp[:,:,0]
    img[:,:,6]=temp[:,:,1]

    temp = computeHSVFeatures(imgHSV)
    #print("HSV")
    img[:,:,7:9]=temp[:,:,0:2]
    img[:,:,10:15]=temp[:,:,2:7]

    temp = computeNIRFeatures(imgRGBN[...,3])
    #print("NIR")
    img[:,:,9]=temp[:,:,0]
    img[:,:,15]=temp[:,:,1]

    return img

def flattenImage(mat): #input must be a 3D array
    flatMat = np.zeros((mat.shape[0]*mat.shape[1],mat.shape[2]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            flatMat[i*mat.shape[0]+j]=mat[i,j]
    return flatMat

def readImageDBN(imagePath):
    img = extractImageFeatures(imagePath)
    img = computeNormal3D(img)
    img = flattenImage(img)
    return img