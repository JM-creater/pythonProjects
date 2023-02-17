#rectangle
'''def rectangle(width, height):

    area = width * height

    print(area)


#width = float(input("Enter number: "))
#height = float(input("Enter number: "))

#rectangle(width, height)
rectangle(12, 8)'''

#triangle
def triangle(a, b, c):

    return (a + b + c) / 2

    area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    #area = math.sqrt(s * (s - a) * (s - b) * (s - c))

    print('The area of the triangle is %.2f' % area)

#a = float(input('Enter length of first side: '))
#b = float(input('Enter length of second side: '))
#c = float(input('Enter length of third side: '))

print(triangle(2, 3, 4))

