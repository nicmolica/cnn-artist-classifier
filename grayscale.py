import os
from PIL import Image

# get input and output folders from the user, verify them
print("Input folder:")
src = input()
print("Output folder:")
dest = input()
if not os.path.isdir(src):
    print("Invalid source folder.")
    exit(1)
elif not os.path.isdir(dest):
    print("Invalid destination folder.")
    exit(1)

# for each file in the input folder, convert it to grayscale and save it to the destination
for file in os.listdir(src):
    if file[-4:] == '.png':
        Image.open(src + "/" + file).convert('LA').save(dest + "/" + file)

print("Successfully converted images to black and white.")