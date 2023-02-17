''''##define a function
#Sample1
def square(x): ##function definition
    return x*x
##define a function
#Sample1
def square(x): ##function definition
    return x*x
print square(4) ##function call
##Sample2
def multiply(x, y=0):
    print("value of x is: ",x)
    print("value of x is: ", y)
    return x*y
print multiply(y=2, x=4)

##Sample3: add two numbers
def add(a, b):
    return a+b
print add(5, 5)

#Sample4: returns absolute value
def absval(x):
    if x < 0:
        return x*-1
    else:
        return x
print absval(-3)

#Sample5:display "Juan Tamad"
def name(a):
    return a
print name("Juan Tamad")'''


#VECTOR RULES & SAMPLES
import numpy as np
##creating a 1-D list (horizontal)
list1 = [1, 2, 3]
##create a vector as a row
vector1 = np.array(list1)
print (vector1)
##create a 1-D list (vertical)
list2 = [[10], [20], [30]]
##create a vecotr as a columns
vector2 = np.array(list2)
print (vector2)

##Vector using arange function
v =  np.arange(0, 5)
print (v)
print (v*2)
##using dot Product
v1 = np.arange(0, 5)
v2 = np.arange(5, 10)
print (v1)
print (v2)
print (np.dot(v1, v2))
print (v1*v2)


#UPDATE OF FUNCTION RULES TO ANSWERS:

def functionname(variable1, variable2):
    return variable1%variable2
functionname(variable1=2, variable2=3)

'''If naa if statements follow this:
if (variablename!=0)
print ("statement")
elif (variablename!=0 operator variablename==0)'''


#Numpy Array Attributes
import numpy as n
n.random.seed(0)##seed for reproducibility
##UsingRandom
a=n.random.randint(10, size=6) #1D array
b=n.random.randint(10, size=(3,4)) #2D Array
c=n.random.randint(10, size=(3,4,5))#3D array
print(c)
#Attributes
#1.ndim->number of dimensions
#2.shape->the size of each dimension
#3.size->total size of the array
#4.dtype->return the data type
#5.itemsize->lists the size(in bytes) of each
#array
#6.nbytes->list the total size(in bytes)
# of the array
print("Number of dimension of c:",c.ndim)
print("Size of each dimensions of c:",c.shape)
print("Total size of c:",c.size)
print("The data type of c:",c.dtype)
print("Item size:",c.itemsize)
print("Total number of bytes:",c.nbytes)
#Array Indexing: Single Elements
#automatic display of numbers
x = n.arange(7)
print(x)
x = n.array([3,4,5,7,8])
print(x)
##Sample Display each values of the index
#1.print(x[0])
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
#1
print(x[::-1])#display in reversed format
print(x[4::-2])#display in reversed format from index
#4
##array Indexing:2D
y = n.array([[6,3,4], [4,8,1]])
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
y_sub[0,1] = 89
print(y_sub)
##Creating copies of an array
z = y_sub[:3,:3].copy()
print(z)
#Set again after copy of an array
z[1,1] = 100
print(z)
#Reshaping of an Array
m = n.arange(1,10).reshape((3,3))
print(m)
m = n.array([5,2,1])
print(m)
print(m.reshape(1,3))
print(m.reshape(3,1))
##Split
m = n.array([1,2,5,6,7,8])
a,b,c = n.split(m,[2,3])
print(a,b,c)
n = n.arange(20).reshape((4,5))
print(n)

##add
print (v1+v2)
##subtract
print (v1-v2)
##2 ways in declaring array containg a range
print (np.array(range(7)))
print (np.arange(0, 7))
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

