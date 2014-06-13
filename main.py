from IPython.terminal.embed import InteractiveShellEmbed
import essentia
from essentia.standard import *
import numpy as np
from pylab import *
import utils

ipshell = InteractiveShellEmbed()

loader = essentia.standard.MonoLoader(filename = 'tom.wav')
audio = loader()

w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()
zcr = ZeroCrossingRate()
dish = DistributionShape()
tctt = TCToTotal()
fft = FFT()


# Data we need to compile
# -----------------------
# ZCR                       - Scalar - Attack + Decay segments
# Kurtosis                  - Scalar - Decay segment
# Skewness                  - Scalar - Decay segment
# Centroid                  - Vector - Decay segment            - 5 orders
# Energy Descriptors (FFT)  - Vector - Attack + Decay segments  - m bands
# MFCC                      - Vector - Attack + Decay segments  - n bands
# ----------------------------------------------------------------
# So we have 4 scalar features and 5 + m + n vector features

pool = essentia.Pool()
attack, decay = utils.extractEnvelopeSegments(audio)

# Spectral centroid
# range = sampleRate / 2
# see http://essentia.upf.edu/documentation/reference/std_CentralMoments.html
cm = CentralMoments(range=22050)
spectralCentroid = cm(spectrum(decay))

# Extracting kurtosis & skewness
ds = DistributionShape()
_, skewness, kurtosis = ds(spectralCentroid)

print skewness
print kurtosis

"""
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
"""
