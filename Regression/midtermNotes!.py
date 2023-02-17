import numpy as np
import numpy.random

np.random.seed(0)


a=np.random.randint(10, size=6) #1D array
b=np.random.randint(10, size=(3,4)) #2D Array
c=np.random.randint(10, size=(3,4,5))#3D array
print(a)
print(b)
print(c)

'''def functionname(variable1, variable2):
    if (variable1 != 0):
        print("statement")
    elif (variable1 != 0 and variable2 == 0):
        print("yawa si miss")
    return variable1%variable2
print(functionname(variable1=0, variable2=3))'''


print("Number of dimension of c:",b.ndim)
print("Size of each dimensions of c:",b.shape)
print("Total size of c:",b.size)
print("The data type of c:",b.dtype)
print("Item size:",a.itemsize)
print("Total number of bytes:",b.nbytes)

x = np.arange(8)
print(x)
x = np.array([3,4,5,7,8])
print(x)

print(x[0])
#1.print(x[2])
print(x[2])
##Samply display last values of the index
##1.print(x[-1])
print(x[-1])
##1.print(x[-2])
print(x[-2])#second last value
print(x[2:5])#sub array
print(x[::2])#display every other element
print(x[1:5])#display every other element from index

print(x[::-2])#display in reversed format
print(x[3::-2])#display in reversed format from index

##array Indexing:2D
                #0         #1
y = np.array([[6,3,4], [4,8,1]])
print(y)
print(y[1,0])
print(y[0,2])
print(y[1,-2])

##Multiply elements by 2
print(y*2)
##Display subarrays for multi array
y_sub = y[:3,:3]
print(y_sub)
##Setting of an array
y_sub[0,2] = 89
print(y_sub)
##Creating copies of an array
z = y_sub[:3,:3].copy()
print(z)
#Set again after copy of an array
z[1,1] = 100
print(z)
#Reshaping of an Array
m = np.arange(1,10).reshape((3,3))
print(m)
m = np.array([5,2,1])
print(m)
print(m.reshape(1,3))
print(m.reshape(3,1))
##Split
m = np.array([1,2,5,6,7,8])
print(m)
a,b,c = np.split(m,[2,4])
print(a,b,c)
n = np.arange(20).reshape((4,5))
print(n)

v1 = np.arange(0, 5)
v2 = np.arange(5, 10)
print(v1)
print(v2)

##add
print (v1+v2)
##subtract
print (v1-v2)
##2 ways in declaring array containg a range
print (np.array(range(7)))
print (np.arange(1, 7))
##Sample1
print (np.arange(8, 20))
##Concatenation of an array
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = np.concatenate([x,y])
print (z)
##hstack -> horizontal stack
z = np.hstack([x, y])
print (z)
##vstack -> vertical stack
z = np.vstack([x, y])
print (z)
##how to use random in a matrix
a = np.random.randint(0, 5, size=(3,3))
print (a)
##how will you multiply the elements by 2
print (a*2)
##how will you multiply th elements of a by itself
print (a*a)
