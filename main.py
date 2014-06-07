from IPython.terminal.embed import InteractiveShellEmbed
import essentia
import essentia.streaming

from essentia.standard import *

ipshell = InteractiveShellEmbed()

loader = essentia.standard.MonoLoader(filename = 'tom.wav')
audio = loader()

from pylab import *

w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()
zcr = ZeroCrossingRate()
dish = DistributionShape()
tctt = TCToTotal()
fft = FFT()


# Data we need to compile:
# ZCR +
# Kurtosis +
# Skewness +
# Centroid +
# Fourier +
# MFCC +


pool = essentia.Pool()

for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
    zero_cross = zcr(spectrum(w(frame)))
    centroid = tctt(spectrum(w(frame)))
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    pool.add('lowlevel.centroid', centroid)
    pool.add('lowlevel.mfcc', mfcc_coeffs)
    pool.add('lowlevel.mfcc_bands', mfcc_bands)

for entry in fft(audio):
    pool.add('fft', float(entry))
pool.add('zcr', zcr(audio))

#imshow(pool['lowlevel.mfcc'].T[1:,:], aspect = 'auto')
# Let's plot mfcc bands on a log-scale so that the energy values will be better 
# differentiated by color
from matplotlib.colors import LogNorm
imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', interpolation = 'nearest', norm = LogNorm())
show()

output = YamlOutput(filename = 'data.sig')
output(pool)
