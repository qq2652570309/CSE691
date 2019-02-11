import numpy as np
import time
import random
'''
# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
'''

z_1 = None
z_2 = None
################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
# Python
z_1 = [[0] * 5] * 3
# NumPy
z_2 = np.zeros((3,5))

#################################################
# 2. Set all the elements in first row of z to 7.
# Python
z_1[0] = [7] * 5
# NumPy
z_2[0, 0:] = 7

#####################################################
# 3. Set all the elements in second column of z to 9.
# Python
for x in z_1: x[1] = 9
# NumPy
z_2[0:, 1] = 9

#############################################################
# 4. Set the element at (second row, third column) of z to 5.
# Python
z_1[1][2] = 5
# NumPy
z_2[1, 2] = 5

##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
x_1 = [m for m in range(50, 100)]
# NumPy
x_2 = np.arange(50,100)

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python
[[x+y for x in range(4)] for y in range(0,16,4)]
# NumPy
y_2 = np.arange(16).reshape((4,4))

##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
# Python
tmp_1 = [[]] * 5
tmp_1[0] = tmp_1[-1] = [1] * 5
tmp_1[1:4] = [1,0,0,0,1]*3

# NumPy
tmp_2 = np.ones((5,5))
tmp_2[1:4,1:4] = 0

##############
print(tmp_1)
print(tmp_2)
##############


a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None
#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python
a_1 = [[x+y for x in range(100)] for y in range(0,5000,100)]
# NumPy
a_2 = np.matrix(np.arange(5000, dtype='int').reshape((50,100)))

###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
b_1 = [[x+y for x in range(200)] for y in range(0,20000,200)]
# NumPy
b_2 = np.matrix(np.arange(20000, dtype='int').reshape((100,200)))

#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
# Python
c_1 = []
for i in range(len(a_1)):
    r = []
    for j in range(len(b_1[0])):
        c = 0
        for k in range(len(a_1[0])):
            c += a_1[i][k] * b_1[k][j]
        r.append(c)
    c_1.append(r)
# NumPy
c_2 = np.dot(a_2, b_2)

d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python
d_1 = [[random.random() for i in range(3)] for j in range(3)]
min1 = d_1[0][0];
max1 = d_1[0][0];
for r in d_1:
    min1 = min(min1, min(r))
    max1 = max(max1, max(r))
for i in range(len(d_1)):
    for j in range(len(d_1[i])):
        d_1[i][j] = (d_1[i][j]-min1)/(max1-min1)
# NumPy
d_2 = np.random.rand(3,3)
min2 = d_2.min()
max2 = d_2.max()
d_2 = (d_2-min2)/(max2-min2)

##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.
# Python
for i in range(len(a_1)):
    mean = sum(a_1[i])/len(a_1[i])
    for j in range(len(a_1[i])):
        a_1[i][j] -= mean
# NumPy
a_2 = a_2 - a_2.mean(1)
# print()

###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
for j in range(200):
    mean = sum([b_1[i][j] for i in range(100)])/100
    for i in range(100):
        b_1[i][j] -= mean
# NumPy
b_2 = b_2 - b_2.mean(0)

################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python
e_1 = []
for j in range(len(c_1[0])):
    e_1.append([c_1[i][j] + 5 for i in range(len(c_1))])
# NumPy
e_2 = np.transpose(c_2) + 5

##################
print (np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python
f_1 = []
for e in e_1:
    f_1 += e
f_1 = np.matrix(f_1)
# NumPy
f_2 = e_2.reshape(e_2.size)

print(f_1.shape)
print(f_2.shape)