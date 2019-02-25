import numpy as np


class Svm (object):
    """" Svm classifier """

    def __init__ (self, inputDim, outputDim):
        self.W = None
        #########################################################################
        # TODO: 5 points                                                        #
        # - Generate a random svm weight matrix to compute loss                 #
        #   with standard normal distribution and Standard deviation = 0.01.    #
        #########################################################################
        self.W = 0.01 * np.random.randn(inputDim, outputDim)

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        Svm loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to weights self.W (dW) with the same shape of self.W.
        """
        loss = 0.0
        dW = np.zeros_like(self.W)
        #############################################################################
        # TODO: 20 points                                                           #
        # - Compute the svm loss and store to loss variable.                        #
        # - Compute gradient and store to dW variable.                              #
        # - Use L2 regularization                                                  #
        # Bonus:                                                                    #
        # - +2 points if done without loop                                          #
        #############################################################################
        
        # batch size is N, number of simpe is K
        N = x.shape[0]
        # calculate the score, S.shape = (N, K)
        s = x.dot(self.W)
        #calculate the score of correct lable, Sy.shape = (N, 1)
        sy = s[np.arange(N), y].reshape(-1, 1)
        # margin = max(0, Sj - Sy + 1), margin.shape = (N, K)
        margin = s - sy + 1
        margin = np.maximum(0, margin)
        margin[np.arange(N), y] = 0
        # calculate the loss
        loss = np.sum(margin) / N
        
        # calculate gradient
        # ds.shape = (N, K)
        ds = np.zeros(s.shape, dtype=np.float)
        # calcuate dSj, j != y
        ds[margin > 0] = 1
        # calculate dSy
        ds[np.arange(N), y] = -np.sum(ds, axis = 1)
        # calculate dW
        dW = x.T.dot(ds) / N
        dW += 2 * reg * self.W
        
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train (self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Svm classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (D, batchSize)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            
            N = np.random.choice(x.shape[0], batchSize)
            xBatch = x[N]
            yBatch = y[N]

            # call lost function
            loss, dW = self.calLoss(xBatch, yBatch, reg)
            # update weight and lossHistory
            self.W = self.W - lr * dW
            lossHistory.append(loss)

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Store the predict output in yPred                                    #
        ###########################################################################

        # calculate score, S.shape = (N, K)
        S = x.dot(self.W)
        # calcuate prediction of y
        yPred = np.argmax(S, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################

        # calculate prediction of y
        yPred = self.predict(x)
        # calculate accuracy
        result = y == yPred
        acc = len(result[result == True]) / result.size
        acc = acc * 100

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



