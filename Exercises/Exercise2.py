def isPrime(n):
    arr = [x for x in range(2,n)]
    for x in arr:
        if (n % x is 0):
            return False
    return True

def prime100():
    print("hello world")
    return [x for x in range(1,100) if isPrime(x)]