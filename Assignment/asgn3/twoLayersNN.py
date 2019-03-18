import numpy as np


class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################
        self.params['w1'] = 0.0001 * np.random.randn(inputDim, hiddenDim)
        self.params['b1'] = np.ones(hiddenDim)
        self.params['w2'] = 0.0001 * np.random.randn(hiddenDim, outputDim)
        self.params['b2'] = np.ones(outputDim)

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################

        N = x.shape[0]

        # Forward pass and calculate score
        x_b = np.hstack((x, np.ones((N,1))))
        u_b = np.vstack((self.params['w1'], self.params['b1']))
        Hin = x_b.dot(u_b)
        Hout = np.maximum(np.zeros(Hin.shape), Hin)
        Hout_b = np.hstack((Hout, np.ones((Hout.shape[0],1))))
        w2_b = np.vstack((self.params['w2'], self.params['b2']))
        s = Hout_b.dot(w2_b)

        # calculate probability
        s = s - np.max(s, axis=1, keepdims=True)
        exp_s = np.exp(s)
        sum_s = np.sum(exp_s, axis=1, keepdims=True)
        prob = exp_s / sum_s

        # calculate loss
        prob_correct = prob[np.arange(N), y]
        loss = np.sum(-np.log(prob_correct)) / N
        loss += 0.5 * reg * (np.sum(self.params['w1']**2) + np.sum(self.params['w2']**2))

        # Backprogagation and calculate gradient
        ind = np.zeros(prob.shape)
        ind[np.arange(N), y] = 1
        ds = prob - ind
        dw2_b = Hout_b.T.dot(ds) / N
        dHout = ds.dot(self.params['w2'].T)
        dHin = np.zeros(dHout.shape)

        for x, y in np.ndindex(Hin.shape):
            if Hin[x, y] > 0:
                dHin[x, y] = dHout[x, y]

        dw1_b = x_b.T.dot(dHin) / N

        grads['w2'] = dw2_b[0:-1] + 2 * reg * self.params['w2']
        grads['b2'] = dw2_b[-1:].reshape(self.params['b2'].shape) + 2 * reg * self.params['b2']
        grads['w1'] = dw1_b[0:-1] + 2 * reg * self.params['w1']
        grads['b1'] = dw1_b[-1:].reshape(self.params['b1'].shape) + 2 * reg * self.params['b1']

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
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
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            R = np.random.choice(x.shape[0], batchSize, replace=True)
            xBatch = x[R]
            yBatch = y[R]
            loss, grads = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)

            self.params['w1'] += -lr * grads['w1']
            self.params['w2'] += -lr * grads['w2']
            self.params['b2'] += -lr * grads['b2']
            self.params['b1'] += -lr * grads['b1']


            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
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
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        # calculate score
        x_b = np.hstack((x, np.ones((x.shape[0],1))))
        u_b = np.vstack((self.params['w1'], self.params['b1']))
        Hin = x_b.dot(u_b)
        Hout = np.maximum(np.zeros(Hin.shape), Hin)
        Hout_b = np.hstack((Hout, np.ones((Hout.shape[0],1))))
        w2_b = np.vstack((self.params['w2'], self.params['b2']))
        s = Hout_b.dot(w2_b)

        yPred = np.argmax(s, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 10 points                                                         #
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



