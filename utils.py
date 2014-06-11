import essentia
from essentia.standard import *
import os
import random
import logging

def sampleGenerator(basedir='drums/', size=1000, seed=666):
    samples = []
    random.seed(seed)
    
    # There is a crazy memleak if you try to create
    # new loader objects in a loop:
    # https://github.com/MTG/essentia/issues/54
    loader = essentia.standard.MonoLoader()

    # This preprocessing sort of defeats the purpose
    # of a generator but w/e
    for dirpath, _, paths in os.walk(basedir):
        samples.extend([dirpath + '/' + path for path in paths])

    i = 0

    while i < size:
        try:
            samplePath = random.choice(samples)
            loader.configure(filename=samplePath)
            sample = loader()
        # Skip if not an audio file
        except RuntimeError:
            logging.warning("Skipping bad file: " + samplePath)
            continue

        yield sample
        i += 1
