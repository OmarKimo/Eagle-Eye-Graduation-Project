import numpy as np
from shapely.wkt import loads
from matplotlib.patches import Polygon
from collections import defaultdict
from shapely.geometry import MultiPolygon, Polygon
import cv2
import random
from ReadFile import ReadTif, ContrastAdjustment
import pandas as pd

#Convert all image coordinates
def ConvertCoord(Coords, Size, XYcoords):
    Xmax, Ymin = XYcoords
    Height, Width=Size
    Width = 1.0 * Width * Width / (Width + 1)
    Height = 1.0 * Height * Height / (Height + 1)
    Xfinal , Yfinal = Width / Xmax , Height / Ymin
    Coords[:, 0] *= Xfinal
    Coords[:, 1] *= Yfinal
    IntCoords = np.round(Coords).astype(np.int32)
    return IntCoords

#Find maximum coordinates for x and minimun coordinates for y
def GetXY(GridSizes, ImageId):
    Xmax, Ymin = GridSizes[GridSizes.ImageId == ImageId].iloc[0, 1:].astype(float)
    return (Xmax, Ymin)

#Find polygon list for class for image
def GetPolygonsList(wkt_list, ImageId, classType):
    Allpolygon = wkt_list[wkt_list.ImageId == ImageId]
    multipolygon = Allpolygon[Allpolygon.ClassType == classType].MultipolygonWKT
    polygonList = None
    if len(multipolygon) != 0:
        polygonList = loads(multipolygon.values[0])
    return polygonList


def GetTheConvertedContours(PolygonList, Size, XYCoods):
    if PolygonList is None:
        return None
    if Size is None or Size[0] <= 0 or Size[1] <= 0:
        return None
    ExteriorList,InteriorList = [],[]
    for Polygon in PolygonList:
        ExteriorList.append(    ConvertCoord( np.array(list(Polygon.exterior.coords)) , Size, XYCoods)  )
        for j in Polygon.interiors:
            InteriorList.append(ConvertCoord(np.array(list(j.coords)), Size, XYCoods))
    return ExteriorList, InteriorList


def PlotMask(Size, Contours):
    if Size is None:
        return None
    if Size[0] <= 0 or Size[1] <= 0:
        return None
    mask = np.zeros(Size, np.uint8)
    if Contours is None:
        return mask
    ExteriorList, InteriorList = Contours
    cv2.fillPoly(mask, ExteriorList, 1)
    cv2.fillPoly(mask, InteriorList, 0)
    return mask

#Fill polygons exterior and interior points and return mask of images
def GenerateMaskForImage(img_size, imageId, class_type, GridSizes, wkt_list):
    XY = GetXY(GridSizes, imageId)
    PolygonList = GetPolygonsList(wkt_list, imageId, class_type)
    contours = GetTheConvertedContours(PolygonList, img_size, XY)
    Mask = PlotMask(img_size, contours)
    return Mask



def MaskToPolygons(mask, epsilon=1, min_area=5):
    if mask is None:
        return MultiPolygon()
    contours, hierarchy = cv2.findContours(((mask == 1) * 255).astype(np.uint8),cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours or hierarchy.shape[0] != 1:
        return MultiPolygon()
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    PolygonsList = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area and cnt.shape[1] == 1:
            temp=[c[:, 0, :] for c in cnt_children.get(idx, []) if cv2.contourArea(c) >= min_area]
            PolygonsList.append(Polygon( shell=cnt[:, 0, :], holes=temp))
    # approximating polygons might have created invalid ones, fix them
    PolygonsList = MultiPolygon(PolygonsList)
    if not PolygonsList.is_valid:
        PolygonsList = PolygonsList.buffer(0)
        if PolygonsList.type == 'Polygon':
            PolygonsList = MultiPolygon([PolygonsList])
    return PolygonsList


NClass = 10  # total number of class
Bands = 3
filepath=""
path ="F:\kaggle_dstl_submission\data"
tr_wkt = pd.read_csv(path+"\\train_wkt_v4.csv")
chosen_patches=2500
Size = 800

def StartTrain():
    #put all image into one image
    x = np.zeros((5 * Size, 5 * Size, Bands)) #input for Unet
    y = np.zeros((5 * Size, 5 * Size, NClass)) # exp output
    ids = sorted(tr_wkt.ImageId.unique())
    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]
            img = ReadTif(id)
            img = ContrastAdjustment(img)
            Xi,Yi=Size * i,Size * j
            x[Xi : Xi + Size, Yi : Yi + Size, :] = img[:Size, :Size,:]
            for Cls in range(NClass):
                y[Xi:Xi + Size, Yi : Yi + Size, Cls] = GenerateMaskForImage((img.shape[0], img.shape[1]),id, Cls + 1) #[:Size, :Size]
    return x,y
    #np.save((filepath + 'data/x_trn_%d') % NClass, x)
    #np.save((filepath + 'data/y_trn_%d') % NClass, y)
