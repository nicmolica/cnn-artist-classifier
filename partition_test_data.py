import os
import random

# get source and destination folders and verify that they're good
print("Source folder for data:")
src = input()
print("Destination fold for test data:")
dest = input()
if not os.path.isdir(src):
    print("Invalid source folder.")
elif not os.path.isdir(dest):
    print("Invalid destination folder.")

# a dictionary mapping artist names to the set of their songs
songs = {
    'chetbaker': set(),
    'billevans': set(),
    'johncoltrane': set(),
    'mccoytyner': set(),
    'bach': set(),
    'mumfordandsons': set(),
    'gregoryalanisakov': set(),
    'mandolinorange': set(),
    'thesteeldrivers': set(),
    'bts': set(),
    'chopin': set(),
    'mamamoo': set(),
    'mozart': set(),
    'seventeen': set(),
    'tchaikovsky': set()
}

# given a file name in the correct format, output the artist's name
def get_artist(file_name):
    segment_and_artist = file_name.split("_")[0]
    if segment_and_artist[1:] in songs:
        artist = segment_and_artist[1:]
    elif segment_and_artist[2:] in songs:
        artist = segment_and_artist[2:]
    elif segment_and_artist[3:] in songs:
        artist = segment_and_artist[3:]
    else:
        print("Invalid file name scheme.")
        exit(1)

    return artist

# for every png in the source directory, add the file to the list of their songs
for file in os.listdir(src):
    if file[-4:] == '.png':
        songs[get_artist(file)].add(file)

# pick a random song from each of the artists and partition it for test
test_partition = []
for artist in songs:
    test_partition.append(random.choice(list(songs[artist])).split("_")[1].split(".")[0])

# move all the partitioned test files to the destination directory
for file in os.listdir(src):
    if file.split("_")[1].split(".")[0] in test_partition:
        os.rename(src + "/" + file, dest + "/" + file)