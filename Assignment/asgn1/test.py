import numpy as np
'''
def isEq(x, y):
    a = np.sum(x == y)
    b = y.size
    return a == b

# 3 * 4
a_1 = [[x+y for x in range(4)] for y in range(0,12,4)]
a_2 = np.arange(12, dtype='int').reshape((3,4))
# 4 * 5
b_1 = [[x+y for x in range(5)] for y in range(0,20,5)]
b_2 = np.arange(20, dtype='int').reshape((4,5))

# print('a:')
# for a in a_1:
#     print(a)
# print(a_2)
# print('b:')
# for b in b_1:
#     print(b)
# print(b_2)


# for i in range(4):
#     for k in range(3):
#         print(a_1[i][k], end=' ')
#     print()
# print()

# for j in range(5):
#     for k in range(3):
#         print(b_1[k][j], end=' ')
#     print()
# print()

c_1 = []
for i in range(3):
    r = []
    for j in range(5):
        c = 0
        for k in range(4):
            c += a_1[i][k] * b_1[k][j]
        r.append(c)
    c_1.append(r)


print("\nc:")
# for c in c_1:
#     print(c)
c_2 = np.dot(a_2, b_2)
# print(c_2)
# print(isEq(c_1, c_2))

# z = sum([a*b for a, b in zip([1,2,3], [4,5,6])])
# print(z)
e_1 = []
for j in range(len(c_1[0])):
    e_1.append([c_1[i][j] + 5 for i in range(len(c_1))])

# print(c_2.size)

e_2 = np.transpose(c_2) + 5
print('\ne:')
for e in e_1:
    print(e)
print()
print(e_2)


print('\nf:')
f_1 = []
for e in e_1:
    f_1 += e
print(np.shape(f_1))

f_2 = e_2.reshape(15)
print(f_2)


x = np.random.rand(3,3)
print(x)
# ((x-min)/(max-min))
min = x.min()
max = x.max()
x = (x-min)/(max-min)
print(x)


import random


d_1 = [[random.random() for i in range(3)] for j in range(3)]
min1 = d_1[0][0];
max1 = d_1[0][0];
for r in d_1:
    min1 = min(min1, min(r))
    max1 = max(max1, max(r))
for i in range(len(d_1)):
    for j in range(len(d_1[i])):
        d_1[i][j] = (d_1[i][j]-min1)/(max1-min1)
print(d_1)


x = np.matrix(np.arange(12).reshape((3, 4)))
print(x)

x = x - x.mean(1)
print(x)
'''



import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [5,4,3,2])
plt.ylabel('some numbers')
plt.show()