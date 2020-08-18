import numpy as np
import tifffile as tiff
import cv2
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K
#################################################### image reordering
K.set_image_data_format('channels_first')   
from common import colorPred

def conv_block(n_neuron, input):
    return Conv2D(n_neuron, 3, activation='relu', padding='same')(input)

def maxPool_block(input):
    return MaxPooling2D(pool_size=(2, 2))(input)

def upSample_block(input1, input2):
    return concatenate([UpSampling2D(size=(2, 2))(input1), input2], axis=1)

class UNetSemanticSegmentation:
    def __init__(self, chosen_bands, save_path="./savedModels/"):
        self.chosen_bands = chosen_bands
        self.save_path = save_path
        self.InputLayerSize = 160
        self.num_classes = 10
        self.chosen_patches = 5000
        self.chosen_case = "" # "esri"
        self.custom_metrics = []
        self.mnSize = 800
        self.nStart_neuron = 32
        self.model = self.buildModel()
        self.model.load_weights(f'{self.save_path}unet_{self.chosen_bands}_10_jk_BestScore')
        
        
    def execute(self, img):
        self.segmentation_masks = self.segment_into_masks(self.model, img)
        self.classes_mask = self.make_general_mask(self.segmentation_masks)
        self.colored_mask = self.add_colors(self.classes_mask)
        return self.classes_mask, self.colored_mask

    def buildModel(self):
        input_layer = Input((self.chosen_bands, self.InputLayerSize, self.InputLayerSize))

        # Contracting Path [Down Sampling]

        conv1_1 = conv_block(self.nStart_neuron, input_layer)
        conv1_2 = conv_block(self.nStart_neuron, conv1_1)
        max_pool1 = maxPool_block(conv1_2)

        conv2_1 = conv_block(self.nStart_neuron * 2, max_pool1)
        conv2_2 = conv_block(self.nStart_neuron * 2, conv2_1)
        max_pool2 = maxPool_block(conv2_2)

        conv3_1 = conv_block(self.nStart_neuron * 4, max_pool2)
        conv3_2 = conv_block(self.nStart_neuron * 4, conv3_1)
        max_pool3 = maxPool_block(conv3_2)

        conv4_1 = conv_block(self.nStart_neuron* 8, max_pool3)
        conv4_2 = conv_block(self.nStart_neuron* 8, conv4_1)
        max_pool4 = maxPool_block(conv4_2)

        conv5_1 = conv_block(self.nStart_neuron * 16, max_pool4)
        conv5_2 = conv_block(self.nStart_neuron * 16, conv5_1)

        # Expansive Path [Up Sampling]

        up_sample6 = upSample_block(conv5_2, conv4_2)
        conv6_1 = conv_block(self.nStart_neuron * 8, up_sample6)
        conv6_2 = conv_block(self.nStart_neuron * 8, conv6_1)

        up_sample7 = upSample_block(conv6_2, conv3_2)
        conv7_1 = conv_block(self.nStart_neuron * 4, up_sample7)
        conv7_2 = conv_block(self.nStart_neuron * 4, conv7_1)

        up_sample8 = upSample_block(conv7_2, conv2_2)
        conv8_1 = conv_block(self.nStart_neuron * 2, up_sample8)
        conv8_2 = conv_block(self.nStart_neuron * 2, conv8_1)

        up_sample9 = upSample_block(conv8_2, conv1_2)
        conv9_1 = conv_block(self.nStart_neuron, up_sample9)
        conv9_2 = conv_block(self.nStart_neuron, conv9_1)

        finalConv = Conv2D(self.num_classes, 1, activation='sigmoid')(conv9_2)

        model = Model(input=input_layer, output=finalConv)
        model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics=['accuracy', *self.custom_metrics])
        
        return model

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
