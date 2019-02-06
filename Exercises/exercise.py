import numpy as np

# Exercise 1: Given the Celsius temperature, convert it to Fahrenheit  (celsius * 1.8 = fahrenheit – 32)
# then check where the temperature in Fahrenheit is more than 75, 
# If yes print “Too hot!”, otherwise print Celsius and Fahrenheit temperature on the screen.

def C2F(c):
    f = c * 1.8 + 32

    if f > 70:
        print('too hot')
    else:
        print(f)

C2F(20)
C2F(30)


# Exercise2: Find the list of prime number from 0-100. 
# Hint: Use List comprehension with function.

def isPrime(n):
    arr = [x for x in range(2,n)]
    for x in arr:
        if (n % x is 0):
            return False
    return True

def prime100():
    print([x for x in range(1,100) if isPrime(x)])

prime100()


# Exercise3: Create a 10x10 array with 0 on the border and 1 inside.

def array1in0():
    a = np.zeros((10,10), dtype='int')
    a[1:9,1:9] = np.ones(8);
    print(a)

array1in0();