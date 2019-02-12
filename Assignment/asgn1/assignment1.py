import numpy as np
import time
import random

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
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

# time data
pythonTime = []
numpyTime = []

z_1 = None
z_2 = None
################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
print("# 1.")
# Python
pythonStartTime = time.time()
z_1 = [[0] * 5] * 3
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
z_2 = np.zeros((3,5))
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

#################################################
# 2. Set all the elements in first row of z to 7.
print("# 2.")
# Python
pythonStartTime = time.time()
z_1[0] = [7] * 5
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
z_2[0, 0:] = 7
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

#####################################################
# 3. Set all the elements in second column of z to 9.
print("# 3.")
# Python
pythonStartTime = time.time()
for x in z_1: x[1] = 9
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
z_2[0:, 1] = 9
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

#############################################################
# 4. Set the element at (second row, third column) of z to 5.
print("# 4.")
# Python
pythonStartTime = time.time()
z_1[1][2] = 5
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
z_2[1, 2] = 5
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
print("\n# 5.")
# Python
pythonStartTime = time.time()
x_1 = [m for m in range(50, 100)]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
x_2 = np.arange(50,100)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
print("\n# 6.")
# Python
pythonStartTime = time.time()
y_1 = [[x+y for x in range(4)] for y in range(0,16,4)]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
y_2 = np.arange(16).reshape((4,4))
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
print("\n# 7.")
# Python
pythonStartTime = time.time()
tmp_1 = [[]] * 5
tmp_1[0] = tmp_1[-1] = [1] * 5
tmp_1[1:4] = [[1,0,0,0,1]]*3
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
tmp_2 = np.ones((5,5))
tmp_2[1:4,1:4] = 0
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

##############
print(tmp_1)
print(tmp_2)
##############


a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None
#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
print("\n# 8.")
# Python
pythonStartTime = time.time()
a_1 = [[x+y for x in range(100)] for y in range(0,5000,100)]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
a_2 = np.matrix(np.arange(5000, dtype='int').reshape((50,100)))
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
print("# 9.")
# Python
pythonStartTime = time.time()
b_1 = [[x+y for x in range(200)] for y in range(0,20000,200)]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
b_2 = np.matrix(np.arange(20000, dtype='int').reshape((100,200)))
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
print("# 10.")
# Python
pythonStartTime = time.time()
c_1 = []
for i in range(len(a_1)):
    r = []
    for j in range(len(b_1[0])):
        c = 0
        for k in range(len(a_1[0])):
            c += a_1[i][k] * b_1[k][j]
        r.append(c)
    c_1.append(r)
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
c_2 = np.dot(a_2, b_2)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
print("# 11.")
# Python
pythonStartTime = time.time()
d_1 = [[random.random() for i in range(3)] for j in range(3)]
min1 = d_1[0][0];
max1 = d_1[0][0];
for r in d_1:
    min1 = min(min1, min(r))
    max1 = max(max1, max(r))
for i in range(len(d_1)):
    for j in range(len(d_1[i])):
        d_1[i][j] = (d_1[i][j]-min1)/(max1-min1)
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
d_2 = np.random.rand(3,3)
min2 = d_2.min()
max2 = d_2.max()
d_2 = (d_2-min2)/(max2-min2)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.
print("\n# 12.")
# Python
pythonStartTime = time.time()
for i in range(len(a_1)):
    mean = sum(a_1[i])/len(a_1[i])
    for j in range(len(a_1[i])):
        a_1[i][j] -= mean
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
a_2 = a_2 - a_2.mean(1)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

###################################################
# 13. Subtract the mean of each column of matrix b.
print("# 13.")
# Python
pythonStartTime = time.time()
for j in range(200):
    mean = sum([b_1[i][j] for i in range(100)])/100
    for i in range(100):
        b_1[i][j] -= mean
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
b_2 = b_2 - b_2.mean(0)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
print("\n# 14.")
# Python
pythonStartTime = time.time()
e_1 = []
for j in range(len(c_1[0])):
    e_1.append([c_1[i][j] + 5 for i in range(len(c_1))])
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
e_2 = np.transpose(c_2) + 5
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 

##################
print (np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
print("\n# 15.")
# Python
pythonStartTime = time.time()
f_1 = []
for e in e_1:
    f_1 += e
f_1 = np.matrix(f_1)
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
f_2 = e_2.reshape(e_2.size)
numPyEndTime = time.time()

print(f_1.shape)
print(f_2.shape)

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
 
 
