import numpy as np
import cv2


def colorPred(num):
    assert num in range(0,11)

    # buildings & manmade structures
    if num < 2: return (255,255,255)
    # Road & Track
    if num < 4: return (255,255,0)
    # Trees & Crops
    if num < 6: return (0,255,0)
    # Waterway & Standing water
    if num < 8: return (0,0, 255)
    # Vehicle Large & Small
    if num < 10: return (0,255,255)
    # unspecified
    if num == 10: return (0,0,0)

#split data to batches for SGD
def batchSplitter(batchSize, x, y=None):
    nBatch = int(np.ceil(len(x) / float(batchSize)))
    #create a random sequence of indices to 'shuffle' x
    permutation = np.random.permutation(len(x))
    
    if y is None:
        xShuffled = x[permutation]
        for i in range(nBatch):
            first = i * batchSize
            last = first + batchSize
            yield xShuffled[first:last]
    else:
        xShuffled = x[permutation]
        yShuffled = y[permutation]
        for i in range(nBatch):
            first = i * batchSize
            last = first + batchSize
            yield xShuffled[first:last], yShuffled[first:last]


def inverseFlatten(pred):
    import matplotlib.pyplot as plt
    assert type(pred) == list
    sz = int(np.sqrt(len(pred)))
    assert sz == 478
    img = np.zeros((sz,sz,3))
    # f = lambda arr: [colorPred(arr[i]) for i in range(len(arr))]
    # img = [f(pred[i*sz : (i+1)*sz]) for i in range(sz)]
    for i in range(sz):
        for j in range(sz):
            img[i][j] = colorPred(pred[i*sz+j])
    img = cv2.resize(img, (800, 800))
    return img
