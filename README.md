# Eagle-Eye
### Eagle Eye is Desktop application for segmentation satellite images using deep learning model U-net.
### this work is based on https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection data 
### main repository is on https://gitlab.com/cufe-cmp4-team11/eagle-eye
### Main objective :
##### 
The core objective of Eagle Eye is to process an input high-resolution satellite image and produce a polygon map that meaningfully describes the image.
Eagle Eye implements two different learning algorithms to realize this objective. The first one involves using the U-Net structure. We also implemented a secondary, more “traditional” Machine Learning algorithm to showcase the advantages of convolutional networks in the task of semantic segmentation. This approach uses the Deep Belief Net structure.
The programming language used in this project is Python. The network structures are implemented using the TensorFlow platform for its high performance and flexibility. The system was tested using the PyTest library.
The outcomes of the project are fairly satisfactory considering the limited computational resources we had to work with. Our best model achieves an accuracy of 80% at 11 label classes.

### System Architecture
![U-net](https://github.com/YahiaAbusaif/Eagle-Eye-Graduation-Project/blob/master/Images/Unet.png)
![U-net](https://github.com/YahiaAbusaif/Eagle-Eye-Graduation-Project/blob/master/Images/U-net%201.png)
![U-net](https://github.com/YahiaAbusaif/Eagle-Eye-Graduation-Project/blob/master/Images/U-net%202.png)


### Result 
this Image is an example of result 
![Input](https://github.com/YahiaAbusaif/Eagle-Eye-Graduation-Project/blob/master/Images/Input.png)
![output](https://github.com/YahiaAbusaif/Eagle-Eye-Graduation-Project/blob/master/Images/Result.png)
