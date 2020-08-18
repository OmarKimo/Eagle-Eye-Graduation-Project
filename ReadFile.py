import numpy as np
import tifffile as tiff
import cv2


def ReadTif(filename, bands=3, size = 800):
    if size==None or bands==None or filename== None:
        return None
    if size<=0:
        return None
    filepath = filename[0:-4] # remove .tif
    dictImgPaths = {
        'RGB': '{}.tif'.format(filepath),      # RGB
        'SWIR': '{}_A.tif'.format(filepath),   # SWIR
        'MULTI': '{}_M.tif'.format(filepath),  # Multispectral
        'PANCH': '{}_P.tif'.format(filepath)   # Panchromatic
    }
    if bands == 3:
        try:
            img = tiff.imread(dictImgPaths['RGB'])
        except:
            print("can't read RGB image")
            return None
        img_rolled = np.rollaxis(img, 0, 3)
        final_img = cv2.resize(img_rolled, (size, size))
        return final_img
    elif bands == 4:
        # for RGB
        try:
            img_RGB = tiff.imread(dictImgPaths['RGB'])
        except:
            print("error can't read RGB image")
            return None
        img_RGB_rolled = np.rollaxis(img_RGB, 0, 3)
        final_img_RGB = cv2.resize(img_RGB_rolled, (size, size))

        # for Multispectral
        try:
            img_M = tiff.imread(dictImgPaths['MULTI'])
        except:
            print("error can't read Multispectral image")
            return None
        
        img_M_adjusted = np.transpose(img_M, (1,2,0))
        final_img_M = cv2.resize(img_M_adjusted, (size, size)) 
        
        img = np.zeros((final_img_RGB.shape[0], final_img_RGB.shape[1], bands), "float32")
        #print(f'RGB shape: {final_img_RGB.shape}\nM shape: {img_M.shape}\nimg shape: {final_img_M.shape}')
        img[..., 0:3] = final_img_RGB
        img[..., 3] = final_img_M[ : , : , 7]
        return img
    elif bands == 20:
        # for RGB 
        try:
            img_RGB = tiff.imread(dictImgPaths['RGB'])
        except:
            print("can't read RGB image")
            return None
        img_RGB_rolled = np.rollaxis(img_RGB, 0, 3)
        final_img_RGB = cv2.resize(img_RGB_rolled, (size, size))

        # for Panchromatic
        try: 
            img_P = tiff.imread(dictImgPaths['PANCH'])
        except:
            print("error can't read Panchromatic image")
            return None
        final_img_P = cv2.resize(img_P, (size, size))

        # for Multispectral
        try:
            img_M = tiff.imread(dictImgPaths['MULTI'])
        except:
            print("error can't read Multispectral image")
            return None
        img_M_adjusted = np.transpose(img_M, (1,2,0))
        final_img_M = cv2.resize(img_M_adjusted, (size, size))

        # for SWIR
        try:
            img_A = tiff.imread(dictImgPaths['SWIR'])
        except:
            print("error can't read SWIR image")
            return None
        img_A_adjusted = np.transpose(img_A, (1,2,0))
        final_img_A = cv2.resize(img_A_adjusted, (size, size))

        img = np.zeros((final_img_RGB.shape[0], final_img_RGB.shape[1], bands), "float32")
        img[..., 0:3] = final_img_RGB
        img[..., 3] = final_img_P
        img[..., 4:12] = final_img_M
        img[..., 12:21] = final_img_A
        return img
    return None



def ContrastAdjustment(img, low_p = 5, high_p = 95):
    img_adjusted = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[2]):
        mn, mx = 0, 1       # np.min(img), np.max(img)
        p_low, p_high = np.percentile(img[:, :, i], low_p), np.percentile(img[:, :, i], high_p)
        tmp = mn + (img[:, :, i] - p_low) * (mx - mn) / (p_high - p_low)
        tmp[tmp < mn] = mn
        tmp[tmp > mx] = mx
        img_adjusted[:, :, i] = tmp
    return img_adjusted.astype(np.float32)