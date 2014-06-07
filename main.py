import essentia
import essentia.streaming

from essentia.standard import *

loader = essentia.standard.MonoLoader(filename = 'tom.wav')
audio = loader()

from pylab import plot, show, figure

w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()

frame = audio[0:100]
spec = spectrum(w(frame))

plot(spec)
show()
