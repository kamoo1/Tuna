import os

with open("neg.txt", "w") as f:
    for filename in os.listdir("bg"):
        f.write("bg/" + filename + "\n")
