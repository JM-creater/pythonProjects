def transform(val):
    num_val = []
    for i in val:
        num_val.append(ord(i))
    sum_item = sum(num_val)
    return sum_item

def transform_1(val):
    a = ""
    for i in range(1, len(val) + 1):
        a += val[len(val) - i]
    return a

string = input("Enter a string: ")

if(string[0] == 'a' or string[0] == 'A' or string[0] == 'e' or string[0] == 'E' or string[0] == 'i' or string[0] == 'I' or string[0] == 'o' or string[0] == 'O' or string[0] == 'u' or string[0] == 'U'):
    print(transform_1(string))
else:
    print(transform(string))