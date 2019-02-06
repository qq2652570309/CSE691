import numpy as np

# 5 * 10
a_1 = [[x+y for x in range(10)] for y in range(0,50,10)]
# 10 * 5
b_1 = [[x+y for x in range(5)] for y in range(0,50,5)]

c_1 = []
for a in a_1:
    t = []
    for b in b_1:
        c = 0
        for i in range(10):
            c += a[i] + b[i]
        t.append(c)
    c_1.append(t)

print(c)