print("1. Calculate the area of the circle")
print("2. Calculate the area of the rectangle")
print("3. Calculate the area of the triangle")
choice = int(input("Enter choice [1-4]: "))
PI = 3.142
def computeareaofcircle(radius):
    area = (PI * (radius * radius))
    return area
def computeareaofrectangle(width, height):
    area = width * height
    return area
def computeareaoftriangle(base, H):
    area = (base * H) / 2
    return area
if choice == 1:
    radius = float(input("Enter the radius: "))
    print("The Area of the circle is %.2f" % computeareaofcircle(radius))
elif choice == 2:
    width = float(input("Enter the width: "))
    height = float(input("Enter the height: "))
    print("The Area of the circle is %.2f" % computeareaofcircle(width))
elif choice == 3:
    base = float(input("Enter the base: "))
    H = float(input("Enter the height: "))
    print("The Area of the triangle is %.2f" % computeareaoftriangle(base, H))
else:
    print("Invalid Choice")





































print("1. Calculate the area of the circle")
print("2. Calculate the area of the rectangle")
print("3. Calculate the area of the triangle")
choice = int(input("Enter choice [1-4]: "))
PI = 3.142
def computeareaofcircle(radius):
    return (PI * (radius * radius))
def computeareaofrectangle(width, height):
    return width * height
def computeareaoftriangle(base, H):
    return (base * H) / 2
if choice == 1:
    radius = float(input("Enter the radius: "))
    area = computeareaofcircle(radius)
    print("The Area of the circle is %.2f" % area)
elif choice == 2:
    width = float(input("Enter the width: "))
    height = float(input("Enter the height: "))
    area = computeareaofrectangle(width, height)
    print("The Area of the circle is %.2f" % area)
elif choice == 3:
    base = float(input("Enter the base: "))
    H = float(input("Enter the height: "))
    area = computeareaoftriangle(base, H)
    print("The Area of the triangle is %.2f" % area)
else:
    print("Invalid Choice")

