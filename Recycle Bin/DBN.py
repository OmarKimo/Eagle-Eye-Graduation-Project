import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
from common import batchSplitter

sess = tf.Session()

class RBM():
    def __init__(self,
                 nHidden=200,
                 learningRate=1e-3,
                 nEpochs=5,
                 nIterCD=1,
                 batchSize=256,
                 verbose=True):
        self.nHidden = nHidden
        self.learningRate = learningRate
        self.nEpochs = nEpochs
        self.nIterCD = nIterCD
        self.batchSize = batchSize
        self.verbose = verbose

    @classmethod
    def variableNames(cls):
        return ['nHidden',
                'nVisible',
                'learningRate',
                'nEpochs',
                'nIterCD',
                'batchSize',
                'verbose']

    # Inititializes an object with values loaded from a file
    @classmethod
    def initFromDict(cls, externalDict):
        weights = {var_name: externalDict.pop(var_name) for var_name in ['W', 'b', 'a']}

        nVisible = externalDict.pop('nVisible')
        newObject = cls(**externalDict)
        setattr(newObject, 'nVisible', nVisible)

        # Initialize RBM
        newObject.defineModel(weights)
        sess.run(tf.variables_initializer([getattr(newObject, name) for name in ['W', 'b', 'a']]))

        return newObject

    # Gets the object's variables to save them in a file
    def saveToDict(self):
        internalDict = {name: self.__getattribute__(name) for name in self.variableNames()}
        internalDict.update(
            {name: self.__getattribute__(name).eval(sess) for name in ['W', 'b', 'a']})
        return internalDict

    def defineModel(self, weights=None):
        # Initialize the weights and biases
        if weights:
            for attr_name, value in weights.items():
                self.__setattr__(attr_name, tf.Variable(value))
        else:            
            std = 1.0 / np.sqrt(self.nVisible)
            self.W = tf.Variable(tf.random_normal([self.nHidden, self.nVisible], std))
            self.b = tf.Variable(tf.random_normal([self.nHidden], std))
            self.a = tf.Variable(tf.random_normal([self.nVisible], std))

        # TensorFlow operations
        # Assign the visible nodes to a placeholder
        self.nodesVisible = tf.placeholder(tf.float32, shape=[None, self.nVisible])
        # Forward Pass - Calculate the hidden nodes: H0 = sigmoid(((W x V^T) + c|b)^T)
        self.calcH0 = tf.nn.sigmoid(tf.transpose(tf.matmul(self.W, tf.transpose(self.nodesVisible))) + self.b)
        # Assign the hidden nodes to a placeholder 
        self.nodesHidden = tf.placeholder(tf.float32, shape=[None, self.nHidden])
        # Backward Pass - Approximate the visible nodes: V' = sigmoid((H x W)+ b|a)
        self.calcV0 = tf.nn.sigmoid(tf.matmul(self.nodesHidden, self.W) + self.a)
        # Create a matrix RUV0 like H with normaly distributed random variables
        random_uniform_values = tf.Variable(tf.random_uniform([self.batchSize, self.nHidden]))
        # Choose sample values Hs0 from H0 using the RUV0
        sampleH0 = tf.cast(random_uniform_values < self.calcH0, 'float32')
        # append the RUV0 to a list of all RUVs
        self.random_variables = [random_uniform_values]

        # we multiply the sample hidden values Hs0 by the visible values V
        # [B, H, 1] x [B, 1, V] = [B, H, V]
        positiveGradient = tf.matmul(tf.expand_dims(sampleH0, 2), tf.expand_dims(self.nodesVisible, 1))

        # Negative gradient
        # Gibbs sampling
        sampleHi = sampleH0
        for i in range(self.nIterCD):
            # V's = sig Hs0 x W + b|a
            calcVi = tf.nn.sigmoid(tf.matmul(sampleHi, self.W) + self.a)
            # H1 = sigmoid(((W x V's^T) + c|b)^T)
            calcHi = tf.nn.sigmoid(tf.transpose(tf.matmul(self.W, tf.transpose(calcVi))) + self.b)
            # create RUVi
            random_uniform_values = tf.Variable(tf.random_uniform([self.batchSize, self.nHidden]))
            # Choose sample values Hs1 from H1 using the RUVi
            sampleHi = tf.cast(random_uniform_values < calcHi, 'float32')
            # append the RUVi
            self.random_variables.append(random_uniform_values)
        
        # [B, H, 1] x [B, 1, V] = [B, H, V]
        negativeGradient = tf.matmul(tf.expand_dims(sampleHi, 2), tf.expand_dims(calcVi, 1))

        # dW = batchAverage(positive - negative)
        calcDeltaW = tf.reduce_mean(positiveGradient - negativeGradient, 0)
        # da = batchAverage(V-V's)
        calcDeltaA = tf.reduce_mean(self.nodesVisible - calcVi, 0)
        # db = batchAverage(Hs0-Hs1)
        calcDeltaB = tf.reduce_mean(sampleH0 - sampleHi, 0)

        # W = W + r*dW
        self.iterateW = tf.assign_add(self.W, self.learningRate * calcDeltaW)
        # a = a + r*da
        self.iterateA = tf.assign_add(self.a, self.learningRate * calcDeltaA)
        # b = b + r*db
        self.iterateB = tf.assign_add(self.b, self.learningRate * calcDeltaB)


    def fit(self, X):
        self.nVisible = X.shape[1]
        self.defineModel()
        sess.run(tf.variables_initializer([self.W, self.b, self.a]))  

        for i in range(self.nEpochs):
            idx = np.random.permutation(len(X))
            data = X[idx]
            for batch in batchSplitter(self.batchSize, data):
                if len(batch) < self.batchSize:
                    # Zero Padding
                    pad = np.zeros((self.batchSize - batch.shape[0], batch.shape[1]), dtype=batch.dtype)
                    batch = np.vstack((batch, pad))
                # Get new random variables
                sess.run(tf.variables_initializer(self.random_variables))
                sess.run([self.iterateW, self.iterateA, self.iterateB],
                         feed_dict={self.nodesVisible: batch})
            if self.verbose:
                print("RBM Epoch", i, "finished.")
        return
    
    # make a forward pass
    def forwardPass(self, X):
        return sess.run(self.calcH0, feed_dict={self.nodesVisible: X})


class DBN():
    def __init__(self,
                 architectureRBM=[100, 100],
                 learningRateNN=1e-3,
                 learningRateRBM=1e-3,
                 nEpochsNN=100,
                 nEpochsRBM=10,
                 nIterCD=1,
                 batchSizeRBM=512,
                 batchSizeNN=32,
                 dropout=0,
                 verbose=True):
        self.architectureRBM = architectureRBM
        self.learningRateRBM = learningRateRBM
        self.nEpochsRBM = nEpochsRBM
        self.nIterCD = nIterCD
        self.batchSizeRBM = batchSizeRBM
        self.stackedRBMs = None
        self.nEpochsNN = nEpochsNN
        self.learningRateNN = learningRateNN
        self.batchSizeNN = batchSizeNN
        self.dropout = dropout
        self.verbose = verbose

    @classmethod
    def variableNames(cls):
        return ['architectureRBM',
                'learningRateRBM',
                'nEpochsRBM',
                'nIterCD',
                'batchSizeRBM',
                'nEpochsNN',
                'learningRateNN',
                'batchSizeNN',
                'dropout',
                'verbose',
                'mapLabel2Index', 
                'mapIndex2Label']

    def save(self, save_path):
        import pickle
        with open(save_path, 'wb') as filePath:
            internalDict = {name: self.__getattribute__(name) for name in self.variableNames()}
            internalDict.update({name: self.__getattribute__(name).eval(sess) for name in ['W', 'b']})
            internalDict['stackedRBMs'] = [rbm.saveToDict() for rbm in self.stackedRBMs]
            internalDict['nClasses'] = self.nClasses
            pickle.dump(internalDict, filePath)

    @classmethod
    def load(cls, load_path):
        import pickle
        with open(load_path, 'rb') as filePath:
            externalDict = pickle.load(filePath)
            weights = {var_name: externalDict.pop(var_name) for var_name in ['W', 'b']}
            nClasses = externalDict.pop('nClasses')
            mapLabel2Index = externalDict.pop('mapLabel2Index')
            mapIndex2Label = externalDict.pop('mapIndex2Label')
            stackedRBMs = externalDict.pop('stackedRBMs')
            
            newObject = cls(**externalDict)
            
            setattr(newObject, 'stackedRBMs', [RBM.initFromDict(rbm) for rbm in stackedRBMs])
            setattr(newObject, 'nClasses', nClasses)
            setattr(newObject, 'mapLabel2Index', mapLabel2Index)
            setattr(newObject, 'mapIndex2Label', mapIndex2Label)
            # Initialize RBM parameters
            newObject.defineModel(weights)
            sess.run(tf.variables_initializer([getattr(newObject, name) for name in ['W', 'b']]))
            return newObject

    # convert class label to mask vector.
    def mapLabels(self, labels, nClasses):
        newLabels = np.zeros([len(labels), nClasses])
        mapLabel2Index, mapIndex2Label = dict(), dict()
        index = 0
        for i, label in enumerate(labels):
            if label not in mapLabel2Index:
                mapIndex2Label[index] = label
                mapLabel2Index[label] = index
                index += 1
            newLabels[i][mapLabel2Index[label]] = 1
        return newLabels, mapLabel2Index, mapIndex2Label

    def defineModel(self, weights=None):
        self.nodesVisible = self.stackedRBMs[0].nodesVisible
        keepProb = tf.placeholder(tf.float32)
        # Apply dropout on the visible nodes
        nodesVisible_drop = tf.nn.dropout(self.nodesVisible, keepProb)
        self.keepProbs = [keepProb]

        # Define tensorflow operation for a forward pass
        self.outputRBM = nodesVisible_drop
        for rbm in self.stackedRBMs:
            self.outputRBM = tf.nn.sigmoid(tf.transpose(tf.matmul(rbm.W, tf.transpose(self.outputRBM))) + rbm.b)
            keepProb = tf.placeholder(tf.float32)
            self.keepProbs.append(keepProb)
            self.outputRBM = tf.nn.dropout(self.outputRBM, keepProb)

        # should be n_nInputNN
        self.nInputNN = self.stackedRBMs[-1].nHidden

        # Initialize the weights and biases
        if weights:
            for attr_name, value in weights.items():
                self.__setattr__(attr_name, tf.Variable(value))
        else:
            std = 1.0 / np.sqrt(self.nInputNN)
            self.W = tf.Variable(tf.random_normal([self.nInputNN, self.nClasses], std))
            self.b = tf.Variable(tf.random_normal([self.nClasses], std))
        
        # Use Stochastic Gradient Descent optimizer and assign the learning rate
        self.optimizerSGD = tf.train.GradientDescentOptimizer(self.learningRateNN)

        # operations
        self.trueY = tf.placeholder(tf.float32, shape=[None, self.nClasses])
        self.predictedY = tf.matmul(self.outputRBM, self.W) + self.b
        self.outputNN = tf.nn.softmax(self.predictedY)
        self.lossFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(tf.stop_gradient(self.trueY), self.predictedY))
        self.trainingStep = self.optimizerSGD.minimize(self.lossFunction)

    def fit(self, X, Y=None):
        #self.pre_train(X)
        self.stackedRBMs = list()
        for nHidden in self.architectureRBM:
            rbm = RBM(nHidden=nHidden,
                            learningRate=self.learningRateRBM,
                            nEpochs=self.nEpochsRBM,
                            nIterCD=self.nIterCD,
                            batchSize=self.batchSizeRBM,
                            verbose=self.verbose)
            self.stackedRBMs.append(rbm)

        # Fit RBM
        if self.verbose:
            print("Unsupervised Learning Phase:")
        inputDataNN = X
        for rbm in self.stackedRBMs:
            rbm.fit(inputDataNN)
            inputDataNN = rbm.forwardPass(inputDataNN)

        # Assign the number of nodes(classes)
        self.nClasses = len(np.unique(Y))
        if self.nClasses == 1:
            Y = np.expand_dims(Y, -1)

        # Build the neural network
        self.defineModel()
        sess.run(tf.variables_initializer([self.W, self.b]))

        # Change given labels to classifier format
        Y, mapLabel2Index, mapIndex2Label = self.mapLabels(Y, self.nClasses)
        self.mapLabel2Index = mapLabel2Index
        self.mapIndex2Label = mapIndex2Label

        if self.verbose:
            print("Supervised Learning Phase:")
        for epoch in range(self.nEpochsNN):
            for batchData, batchLabels in batchSplitter(self.batchSizeNN, X, Y):
                feed_dict = {self.nodesVisible: batchData, self.trueY: batchLabels}
                feed_dict.update({placeholder: (1 - self.dropout) for placeholder in self.keepProbs})
                sess.run(self.trainingStep, feed_dict=feed_dict)

            if self.verbose:
                print("NN Epoch %d finished." % (epoch))
        if self.verbose:
            print("Model finished.")
        return self

    def predict(self, X):
        # Predict probability of each classes for every datapoint
        feed_dict = {self.nodesVisible: X}
        feed_dict.update({placeholder: 1.0 for placeholder in self.keepProbs})
        probs = sess.run(self.outputNN, feed_dict=feed_dict)
        indexes = np.argmax(probs, axis=1)
        # Change network output to given labels
        labels = map(lambda idx: self.mapIndex2Label[idx], indexes)
        return list(labels)

