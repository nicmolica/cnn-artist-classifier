import os
import time
import numpy as np
import librosa
import librosa.display
import threading
import matplotlib.pyplot as plt
from PIL import Image
from pydub import AudioSegment

# make sure user is aware of format requirements for audio files, get source/dest folder and verify them
print("Please ensure that all audio files are in wav PCM 8-bit format with sampling rate of 8000kHz, and mono rather than multi-channel.")
print("Specify source folder for audio files:")
src = input()
print("Specify destination folder for training/testing data:")
dest = input()
if not os.path.isdir(src) or not os.path.isdir(dest):
    print("Both the source and the destination must be directories.")
    exit(1)

# make temporary staging directories in the destination folder
clipped_audio = dest + "/clipped_audio"
os.mkdir(clipped_audio)
plotted_clips = dest + "/plotted_clips"
os.mkdir(plotted_clips)

# set important global vars for segmenting audio and cropping final pngs
segment_size = 30 * 1000 # 30 second audio clips
left, right, top, bottom = 6, 779, 22, 76 # cropping indices for plotted audio

# Split up the audio into 30-second interleaving segments, where each
# consecutive segment jumps forward by 1/3 of the segment size.
def split_audio(audio):
    segments = []
    top = 0
    bottom = segment_size
    while bottom < len(audio):
        segments.append(audio[top:bottom])
        top += segment_size / 3
        bottom += segment_size / 3
    return segments

# export the split audio segments into the staging area for the next step (plotting)
def export_audio_segments(prefix, pieces):
    i = 0
    for p in pieces:
        p.export(clipped_audio + "/" + str(i + 1) + prefix, format='wav')
        i += 1

# plot a 30-second audio file as a mel spectrogram
def plot_audio(path):
	y, sr = librosa.load(path)
	spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
	spect = librosa.power_to_db(spect, ref=np.max)
	plt.figure(figsize=(10,1))
	librosa.display.specshow(spect, fmax=8000)
	plt.savefig(plotted_clips + "/" + path.split("/")[-1].split(".")[0] + ".png", bbox_inches='tight', pad_inches=0)
	plt.close()

# crop a mel spectrogram to only include the spectrogram itself
def crop(path):
    im = Image.open(plotted_clips + "/" + path)
    im.crop((left, top, right, bottom)).save(dest + "/" + path, quality = 100)

# the target function for the thread responsible for segmenting the audio into 30-second segments
def segmenting_thread_func():
    for file in os.listdir(src):
        if file[-4:] == '.wav':
            export_audio_segments(file, split_audio(AudioSegment.from_wav(src + "/" + file)))
    print("Finished segmenting audio.")

# the target function for the thread responsible for plotting the audio as a mel spectrogram
def plotting_thread_func():
    while True:
        for file in os.listdir(clipped_audio):
            if file[-4:] == '.wav':
                plot_audio(clipped_audio + "/" + file)
                os.remove(clipped_audio + "/" + file)

        # pause for 2 seconds in case we're waiting on another thread, break if more files don't appear
        time.sleep(2)
        try:
            os.remove(clipped_audio + "/" + ".DS_Store")
        except FileNotFoundError:
            pass
        if len(os.listdir(clipped_audio)) == 0:
            break
    print("Finished plotting audio.")

# the target function for the thread responsible for cropping each mel spectrogram
def cropping_thread_func():
    while True:
        for file in os.listdir(plotted_clips):
            if file[-4:] == '.png':
                crop(file)
                os.remove(plotted_clips + "/" + file)

        # pause for 5 seconds in case we're waiting on another thread, break if more files don't appear
        time.sleep(5)
        try:
            os.remove(plotted_clips + "/" + ".DS_Store")
        except FileNotFoundError:
            pass
        if len(os.listdir(plotted_clips)) == 0:
            break
    print("Finished cropping images.")

# initialize threads for segmenting and cropping
segmenter = threading.Thread(target=segmenting_thread_func)
cropper = threading.Thread(target=cropping_thread_func)

# start the two threads
segmenter.start()
cropper.start()

# do all plotting in the main thread (this is a requirement of the matplotlib module)
plotting_thread_func()

# wait for all threads to finish
segmenter.join()
cropper.join()

# remove the staging directories
os.rmdir(clipped_audio)
os.rmdir(plotted_clips)