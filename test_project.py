import pytest
from preprocess import GetXY,GetPolygonsList,PlotMask,GetTheConvertedContours,MaskToPolygons
import numpy as np
import pandas as pd
from ReadFile import ReadTif


path1 ="F:\kaggle_dstl_submission\data"#input("enter the path to file train_wkt_v4.csv ")
tr_wkt = pd.read_csv(path1+"\\train_wkt_v4.csv")  # Well-known text Multipolygon
path2 ="F:\kaggle_dstl_submission\data"#input("enter the path to file grid_sizes.csv ")
grid_sizes = pd.read_csv(path2+"\\grid_sizes.csv", names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


class unitetest:
    def test_GetXY(self):
        x,y = GetXY(GridSizes=grid_sizes, ImageId="6010_0_0")
        nx,ny = 0.009188,-0.009040
        assert abs(x-nx) <=1e-6 ,"GetXY test failed"
        assert abs(y-ny) <=1e-6 ,"GetXY test failed"
        x,y = GetXY(GridSizes=grid_sizes, ImageId="6010_0_1")
        nx,ny = 0.009169,-0.009042
        assert abs(x-nx) <=1e-6 ,"GetXY test failed"
        assert abs(y-ny) <=1e-6 ,"GetXY test failed"
        x,y = GetXY(GridSizes=grid_sizes, ImageId="6060_1_3")
        nx,ny = 0.009158,-0.009043
        assert abs(x-nx) <=1e-6 ,"GetXY test failed"
        assert abs(y-ny) <=1e-6 ,"GetXY test failed"

    def test_GetPolygonsList(self):
        x = GetPolygonsList(wkt_list=tr_wkt, ImageId="6040_2_2",classType=2)
        y = 0
        assert len(x) == y, "GetPolygonsList failed"
        x = GetPolygonsList(wkt_list=tr_wkt, ImageId="6040_2_2", classType=5)
        y = 3879
        assert len(x) == y, "GetPolygonsList failed"
        x = GetPolygonsList(wkt_list=tr_wkt, ImageId="", classType=2)
        y = None
        assert x == y, "GetPolygonsList failed"
        x = GetPolygonsList(wkt_list=tr_wkt, ImageId="6040_2_2", classType=200)
        y = None
        assert x == y, "GetPolygonsList failed"
        x = GetPolygonsList(wkt_list=tr_wkt, ImageId="6040_2_2", classType=-8)
        y = None
        assert x == y, "GetPolygonsList failed"
        x = GetPolygonsList(wkt_list=tr_wkt, ImageId="6040_2_2", classType=1.87)
        y = None
        assert x == y, "GetPolygonsList failed"
        x = GetPolygonsList(wkt_list=tr_wkt, ImageId="6160_2_1", classType=4)
        y = 8
        assert len(x) == y, "GetPolygonsList failed"

    def test_PlotMask(self):
        x = PlotMask(Size=(10,10), Contours=None)
        assert x.all() == 0, "PlotMask test failed"
        x = PlotMask(Size=(-2,10), Contours=None)
        assert x == None, "PlotMask test failed"
        x = PlotMask(Size=(1,0), Contours=None)
        assert x == None, "PlotMask test failed"
        x = PlotMask(Size=None, Contours=None)
        assert x == None, "PlotMask test failed"

    def test_GetTheConvertedContours(self):
        x = GetTheConvertedContours(PolygonList=None, Size=(-2,10), XYCoods=None)
        assert x == None, "GetTheConvertedContours test failed"
        xy = GetXY(GridSizes=grid_sizes, ImageId="6040_2_2")
        x = GetTheConvertedContours(PolygonList=None, Size=(-2,10), XYCoods=xy)
        assert x == None, "GetTheConvertedContours test failed"
        polygons = GetPolygonsList(wkt_list=tr_wkt, ImageId="6040_2_2", classType=5)
        x = GetTheConvertedContours(PolygonList=polygons, Size=(-2,10), XYCoods=xy)
        assert x == None, "GetTheConvertedContours test failed"
        x = GetTheConvertedContours(PolygonList=polygons, Size=(800,800), XYCoods=xy)
        assert x != None, "get_and_convert_contours test failed"


    def test_ReadTif(self):
        x = ReadTif(filename=None,dims=None,size = 800)
        assert x == None, "ReadTif test failed"
        x = ReadTif(filename=None, dims=None, size=-300)
        assert x == None, "ReadTif test failed"
        x = ReadTif(filename=None, dims=3, size=0)
        assert x == None, "ReadTif test failed"
        x = ReadTif(filename="F:\\kaggle_dstl_submission\\data\\sixteen_band\\6010_0_0.tif", dims=3, size=800)
        assert x.any , "ReadTif test failed"

    def test_MaskToPolygons(self):
        size=(10,10)
        mask = np.zeros(size, np.uint8)
        x = MaskToPolygons(mask, epsilon=1, min_area=10)
        assert x.is_empty, "MaskToPolygons test failed"
        mask = np.ones(size, np.uint8)
        x = MaskToPolygons(mask, epsilon=1, min_area=10)
        assert not x.is_empty, "MaskToPolygons test failed"
        mask = np.ones((2,2), np.uint8)
        x = MaskToPolygons(mask, epsilon=1, min_area=10)
        assert x.is_empty, "MaskToPolygons test failed"
        x = MaskToPolygons(mask, epsilon=1, min_area=500)
        assert x.is_empty, "MaskToPolygons test failed"
        mask = np.array([ [0,1,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,1,0,0,0,0,0,0],
                          [0,0,0,1,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0]],np.uint8)
        x = MaskToPolygons(mask, epsilon=1, min_area=4)
        assert not x.is_empty , "MaskToPolygons test failed"
        mask = np.array([ [0,1,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,1,0,0,0,0,1,0],
                          [0,0,0,1,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,1,1,0],
                          [0,0,0,0,0,0,0,1,1,0],
                          [0,0,0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,1,0]],np.uint8)
        x = MaskToPolygons(mask, epsilon=1, min_area=1)
        assert not x.is_empty , "MaskToPolygons test failed"
        mask = np.array([ [0,1,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,1,0,0,0,0,1,0],
                          [0,0,0,1,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,1,1,0],
                          [0,0,0,0,0,0,0,1,1,0],
                          [0,0,0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,1,0]],np.uint8)
        x = MaskToPolygons(mask, epsilon=1, min_area=5)
        assert  x.is_empty , "MaskToPolygons test failed"
        mask = np.array([ [0,1,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,1,0,0,0,0,1,0],
                          [0,0,0,1,0,0,0,1,1,0],
                          [0,0,0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,1,1,0]],np.uint8)
        x = MaskToPolygons(mask, epsilon=1, min_area=5)
        assert  not x.is_empty , "MaskToPolygons test failed"



print("this test done for input and output for csv file  ")
test=unitetest()
test.test_GetXY()
print("GetXY test passed")
test.test_GetPolygonsList()
print("GetPolygonsList test passed")
test.test_PlotMask()
print("PlotMask test passed")
test.test_GetTheConvertedContours()
print("GetTheConvertedContours test passed")
test.test_ReadTif()
print("ReadTif test passed")
test.test_MaskToPolygons()
print("MaskToPolygons test passed")

print("GUI, model, main ,detection ,readfile had been tested manuly")
