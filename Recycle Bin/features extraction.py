import numpy as np
import tifffile as tiff
import math
import cmath

def size(polygon):
    cnt=0
    for i in polygon:
        cnt=cnt+1

def mean(polygon):
    num=size(polygon)
    #not sure if 1,2,3 or 2,3,4
    band1 = polygon.GetRasterBand(1)
    band2 = polygon.GetRasterBand(2)
    band3 = polygon.GetRasterBand(3)
    red = band1.ReadAsArray()
    green = band2.ReadAsArray()
    blue = band3.ReadAsArray()
    Mean,R,G,B=0,0,0,0
    for i in polygon:
        Mean= mean+i
    for i in red:
        R=R+i
    for i in green:
        G=G+i
    for i in blue:
        B=B+i
    Mean=Mean/num
    R=R/num
    G=G/num
    B=B/num
    return Mean,R,G,B

def entropy(polygon):
    lensig = size(polygon)
    symset = list(set(polygon))
    numsym = len(symset)
    propab = [np.size(polygon[polygon == i]) / (1.0 * lensig) for i in symset]
    ent = np.sum([p * np.log2(1.0 / p) for p in propab])
    return ent



def extract(polygon):
    features=[]
    features.append(mean(polygon))
    features.append(entropy(polygon))



    return features





