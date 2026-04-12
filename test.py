import os

path = "C:\\Users\\thien\\Desktop\\N\\Games"
for r, d, fs in os.walk(path):
    for f in fs:
        cake = os.path.join(r, f)
        print(cake)


    