N = int(input("Total number of characters: "))
val = []
count = 1

for i in range(1, N + 1):
    ele = str(input(f"Enter character {i}: "))
    val.append(ele)
    count += 1


def findMaxChar(r):
    item = []
    len(r)
    for x in r:
        item.extend(ord(n) for n in x)
        item.sort()
    return chr(item[-1])


print("Max:", findMaxChar(val))




