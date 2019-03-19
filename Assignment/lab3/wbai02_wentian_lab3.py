import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def lrelu (tmp):
  return tf.nn.relu(tmp) - 0.01 * tf.nn.relu(-tmp)

# Build Softmax classifier same as in Homework 3
from keras.datasets import cifar10
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Mean Image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Reshape data from channel to rows
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))

# Add bias dimension columns
xTrain = np.hstack([xTrain, np.ones((xTrain.shape[0], 1))])
xVal = np.hstack([xVal, np.ones((xVal.shape[0], 1))])
xTest = np.hstack([xTest, np.ones((xTest.shape[0], 1))])

# Build graph
x = tf.placeholder(tf.float32, shape=[None, 3073])
w1 = tf.Variable(tf.random_normal(mean=0.0, stddev=0.01, shape=[xTrain.shape[1], 100]))
b1 = tf.Variable(tf.ones(100))
w2 = tf.Variable(tf.random_normal(mean=0.0, stddev=0.01, shape=[100, 10]))
b2 = tf.Variable(tf.ones(10))
y = tf.placeholder(tf.int64, [None])

# Calculate score
Hin = tf.add(tf.matmul(x, w1), b1)
Hout = lrelu(Hin)
score = tf.add(tf.matmul(Hout, w2), b2)
score = lrelu(score)

# Calculate loss
meanLoss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), logits=score)) + 5e-3*tf.nn.l2_loss(w1) + 5e-3*tf.nn.l2_loss(w2)

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(5e-3)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999)
trainStep = optimizer.minimize(meanLoss)

# Define correct Prediction and accuracy
correctPrediction = tf.equal(tf.argmax(score, 1), y)
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))*100

# Create Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
startTime = time.time()
for i in range(1500):
    # Mini batch
    batchID = np.random.choice(xTrain.shape[0], 1000, replace=True)
    xBatch = xTrain[batchID]
    yBatch = yTrain[batchID]

    # Train
    loss, acc, _ = sess.run([meanLoss, accuracy, trainStep], feed_dict={x: xBatch, y: yBatch})

    if i % 100 == 0:
        print('Loop {0} loss {1}'.format(i, loss))

# Print all accuracy
print ('Training time: {0}'.format(time.time() - startTime))
print ('Training acc:   {0}%'.format(sess.run(accuracy, feed_dict={x: xTrain, y: yTrain})))
print ('Validating acc: {0}%'.format(sess.run(accuracy, feed_dict={x: xVal, y: yVal})))
print ('Testing acc:    {0}%'.format(sess.run(accuracy, feed_dict={x: xTest, y: yTest})))




