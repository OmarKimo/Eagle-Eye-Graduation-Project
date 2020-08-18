import numpy as np
import tifffile as tiff
import cv2
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_first')
from common import colorPred

  
class UNetSemanticSegmentation:
    def __init__(self, chosen_bands, img_RGB_filename, save_path="./savedModels/"):
        self.chosen_bands = chosen_bands
        self.img_path = img_RGB_filename
        self.save_path = save_path
        self.InputLayerSize = 160
        self.num_classes = 10
        self.chosen_patches = 5000
        self.chosen_case = "" # "esri"
        self.custom_metrics = []
        self.mnSize = 800
        self.dictImages = {
            'RGB': '{}.tif'.format(self.img_path),      # RGB
            'SWIR': '{}_A.tif'.format(self.img_path),   # SWIR
            'MULTI': '{}_M.tif'.format(self.img_path),  # Multispectral
            'PANCH': '{}_P.tif'.format(self.img_path)   # Panchromatic
        }
        
    def execute(self):
        self.testImage = self.construct_image()
        self.adjustedImage = self.ContrastAdjustment(self.testImage)
        self.model = self.buildModel()
        self.model.load_weights(f'{self.save_path}unet_{self.chosen_bands}_10_jk_BestScore')
        self.segmentation_masks = self.segment_into_masks(self.model, self.adjustedImage)
        self.classes_mask = self.make_general_mask(self.segmentation_masks)
        self.colored_mask = self.add_colors(self.classes_mask)
        return self.adjustedImage, self.colored_mask

    def ContrastAdjustment(self, img, low_p = 5, high_p = 95):    # https://www.kaggle.com/aamaia/rgb-using-m-bands-example
        img_adjusted = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            mn, mx = 0, 1       # np.min(img), np.max(img)
            p_low, p_high = np.percentile(img[:, :, i], low_p), np.percentile(img[:, :, i], high_p)
            tmp = mn + (img[:, :, i] - p_low) * (mx - mn) / (p_high - p_low)
            tmp[tmp < mn] = mn
            tmp[tmp > mx] = mx
            img_adjusted[:, :, i] = tmp
        return img_adjusted.astype(np.float32)

    def buildModel(self):
        input_layer = Input((self.chosen_bands, self.InputLayerSize, self.InputLayerSize))

        # Contracting Path [Down Sampling]

        conv1_1 = Conv2D(32, 3, activation='relu', padding='same')(input_layer)
        conv1_2 = Conv2D(32, 3, activation='relu', padding='same')(conv1_1)
        max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

        conv2_1 = Conv2D(64, 3, activation='relu', padding='same')(max_pool1)
        conv2_2 = Conv2D(64, 3, activation='relu', padding='same')(conv2_1)
        max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

        conv3_1 = Conv2D(128, 3, activation='relu', padding='same')(max_pool2)
        conv3_2 = Conv2D(128, 3, activation='relu', padding='same')(conv3_1)
        max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

        conv4_1 = Conv2D(256, 3, activation='relu', padding='same')(max_pool3)
        conv4_2 = Conv2D(256, 3, activation='relu', padding='same')(conv4_1)
        max_pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

        conv5_1 = Conv2D(512, 3, activation='relu', padding ='same')(max_pool4)
        conv5_2 = Conv2D(512, 3, activation='relu', padding ='same')(conv5_1)

        # Expansive Path [Up Sampling]

        up_sample6 = concatenate([UpSampling2D(size=(2, 2))(conv5_2), conv4_2], axis=1)
        conv6_1 = Conv2D(256, 3, activation='relu', padding='same')(up_sample6)
        conv6_2 = Conv2D(256, 3, activation='relu', padding='same')(conv6_1)

        up_sample7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3_2], axis=1)
        conv7_1 = Conv2D(128, 3, activation='relu', padding='same')(up_sample7)
        conv7_2 = Conv2D(128, 3, activation='relu', padding='same')(conv7_1)

        up_sample8 = concatenate([UpSampling2D(size=(2, 2))(conv7_2), conv2_2], axis=1)
        conv8_1 = Conv2D(64, 3, activation='relu', padding='same')(up_sample8)
        conv8_2 = Conv2D(64, 3, activation='relu', padding='same')(conv8_1)

        up_sample9 = concatenate([UpSampling2D(size=(2, 2))(conv8_2), conv1_2], axis=1)
        conv9_1 = Conv2D(32, 3, activation='relu', padding='same')(up_sample9)
        conv9_2 = Conv2D(32, 3, activation='relu', padding='same')(conv9_1)

        finalConv = Conv2D(self.num_classes, 1, activation='sigmoid')(conv9_2)

        model = Model(input=input_layer, output=finalConv)
        model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics=['accuracy', *self.custom_metrics])
        
        return model

    def construct_image(self):
        if self.chosen_bands == 3:
            if self.chosen_case == "esri":
                dir = "/content/drive/My Drive/GP/Raster/"
                pattern = r'*.tif'
                gen = glob.iglob(os.path.join(dir, pattern))
                filename = next(gen)
                img_4 = tiff.imread(filename)
                img = np.zeros((3000, 3000, dims), "float32")
                img = img_4[ 1500:2500 , 5000:6000 , 0:3]
                #img = np.rollaxis(img, 0, 3)
                img = cv2.resize(img, (size, size))
                return img
            else:
                img = tiff.imread(self.dictImages['RGB'])
                img_rolled = np.rollaxis(img, 0, 3)
                final_img = cv2.resize(img_rolled, (self.mnSize, self.mnSize))
                return final_img
        
        elif self.chosen_bands  == 4:
            if self.chosen_case == 'esri':
                dir = "/content/drive/My Drive/GP/Raster/"
                pattern = r'*.tif'
                gen = glob.iglob(os.path.join(dir, pattern))
                filename = next(gen)
                img = tiff.imread(filename)
                #img = cv2.resize(img, (size, size))
                return img
            else:
                # RGB 
                img_RGB = tiff.imread(self.dictImages['RGB'])
                img_RGB_rolled = np.rollaxis(img_RGB, 0, 3)
                final_img_RGB = cv2.resize(img_RGB_rolled, (self.mnSize, self.mnSize))
                
                # Multispectral
                img_M = tiff.imread(self.dictImages['MULTI'])
                img_M_adjusted = np.transpose(img_M, (1,2,0))
                final_img_M = cv2.resize(img_M_adjusted, (self.mnSize, self.mnSize))                
                
                img = np.zeros((final_img_RGB.shape[0], final_img_RGB.shape[1], self.chosen_bands), "float32")
                #print(f'RGB shape: {final_img_RGB.shape}\nM shape: {img_M.shape}\nimg shape: {final_img_M.shape}')
                img[..., 0:3] = final_img_RGB
                img[..., 3] = final_img_M[ : , : , 7]
                return img
            
        elif self.chosen_bands == 20:
            # RGB 
            img_RGB = tiff.imread(self.dictImages['RGB'])
            img_RGB_rolled = np.rollaxis(img_RGB, 0, 3)
            final_img_RGB = cv2.resize(img_RGB_rolled, (self.mnSize, self.mnSize))

            # Panchromatic
            img_P = tiff.imread(self.dictImages['PANCH'])
            final_img_P = cv2.resize(img_P, (self.mnSize, self.mnSize))

            # Multispectral
            img_M = tiff.imread(self.dictImages['MULTI'])
            img_M_adjusted = np.transpose(img_M, (1,2,0))
            final_img_M = cv2.resize(img_M_adjusted, (self.mnSize, self.mnSize))

            # SWIR
            img_A = tiff.imread(self.dictImages['SWIR'])
            img_A_adjusted = np.transpose(img_A, (1,2,0))
            final_img_A = cv2.resize(img_A_adjusted, (self.mnSize, self.mnSize))

            img = np.zeros((final_img_RGB.shape[0], final_img_RGB.shape[1], self.chosen_bands), "float32")
            img[..., 0:3] = final_img_RGB
            img[..., 3] = final_img_P
            img[..., 4:12] = final_img_M
            img[..., 12:21] = final_img_A
            return img
        return None

    def segment_into_masks(self, model, img):
        thresholds = np.load(f'{self.save_path}unet_{self.chosen_bands}_thresholds.npy')

        cnv = np.zeros((960, 960, self.chosen_bands)).astype(np.float32)
        prd = np.zeros((self.num_classes, 960, 960)).astype(np.float32)
        cnv[:img.shape[0], :img.shape[1], :] = img

        for i in range(0, 6):
            line = [cnv[i * self.InputLayerSize:(i + 1) * self.InputLayerSize, j * self.InputLayerSize:(j + 1) * self.InputLayerSize] for j in range(6)]
            nw_img = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
            tmp = model.predict(nw_img, batch_size=4)
            for j in range(tmp.shape[0]):
                prd[:, i * self.InputLayerSize:(i + 1) * self.InputLayerSize, j * self.InputLayerSize:(j + 1) * self.InputLayerSize] = tmp[j]
        
        for i in range(self.num_classes):
            prd[i] = prd[i] > thresholds[i]

        return prd[:, :img.shape[0], :img.shape[1]]
    
    def make_general_mask(self, masks):
        n = len(masks)
        assert n == self.num_classes
        ret_mask = np.zeros((masks.shape[1], masks.shape[2])).astype(np.float32)
        for x in range(masks.shape[1]):
            for y in range(masks.shape[2]):
                idx = 10
                for i in range(n):
                    if masks[i][x][y]:
                        idx = i
                        break
                ret_mask[x][y]= idx
        return ret_mask

    def add_colors(self, mask):
        #f = lambda arr: [[colorPred(arr[i][j]) for i in range(arr.shape[0])] for j in range(arr.shape[1])]
        img = np.zeros((mask.shape[0], mask.shape[1], 3))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                img[i][j] = colorPred(mask[i][j])
        return img
