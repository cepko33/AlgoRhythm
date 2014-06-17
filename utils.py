import essentia
from essentia.standard import *
import os
import random
import logging

# In order to perform fft, the size of the array must be even
def ensureEven(audio):
    if len(audio) % 2 != 0:
        audio = audio[:-1]
    return audio

# To divide the audio into attack and decay segments, split along the
# global maximum amplitude.
def extractEnvelopeSegments(audio):
    pd = PeakDetection(orderBy='amplitude')
    duration = Duration()
    midpoint, _ = pd(audio)
    slicer = Slicer(startTimes=essentia.array([0, midpoint[0]*duration(audio)]),
                    endTimes=essentia.array([midpoint[0]*duration(audio), duration(audio)]))
    slices = slicer(audio)

    # XXX: ugly
    return ensureEven(slices[0]), ensureEven(slices[1])

def sampleGenerator(basedir='drums/'):
    samples = []
    
    # There is a crazy memleak if you try to create
    # new loader objects in a loop:
    # https://github.com/MTG/essentia/issues/54
    loader = essentia.standard.MonoLoader()

    # This preprocessing sort of defeats the purpose
    # of a generator but w/e
    for dirpath, _, paths in os.walk(basedir):
        samples.extend([dirpath + '/' + path for path in paths])

    while len(samples) > 0:       
        try:
            samplePath = samples.pop()
            loader.configure(filename=samplePath)
            sample = loader()
        # Skip if not an audio file
        except RuntimeError:
            logging.warning("Skipping bad file: " + samplePath)
            continue

        yield sample
