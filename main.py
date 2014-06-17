from IPython.terminal.embed import InteractiveShellEmbed
import essentia
from essentia.standard import *
from pylab import *
import utils
import sys
from sklearn import neighbors
import numpy as np

try:
    import config
except ImportError:
    print "Please make a config.py file, see config.py.example"
    sys.exit(1)

def poolToNumpy(pool):
    result = []
    for key in pool.descriptorNames():
        result.append(pool[key])
    return np.array(result).flatten()
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

def computeFeatures(pool, sample):
    attack, decay = utils.extractEnvelopeSegments(sample)
    decay_spectrum = spectrum(decay)
    # ZCR
    pool.add('zcr.attack', zcr(attack))
    pool.add('zcr.decay', zcr(decay))
    # Centroid XXX: scalarize
    spectralCentroid = cm(decay_spectrum)
    pool.add('centroid', spectralCentroid[0])
    # Skewness/kurtosis
    _, skewness, kurtosis = ds(spectralCentroid)
    pool.add('skewness', skewness)
    pool.add('kurtosis', kurtosis)
    # MFCC ignoring bands XXX: scalarize
    _, mfcc_coeffs = mfcc(spectrum(utils.ensureEven(sample)))
    for i, coeff in enumerate(mfcc_coeffs):
        pool.add('mfcc_coeffs_' + str(i), coeff)
    # FFT bands, using ranges from paper
    pool.add('energy_40_70', energy_40_70(decay_spectrum))
    pool.add('energy_70_110', energy_70_110(decay_spectrum))
    pool.add('energy_130_145', energy_130_145(decay_spectrum))
    pool.add('energy_160_190', energy_160_190(decay_spectrum))
    pool.add('energy_300_400', energy_300_400(decay_spectrum))
    pool.add('energy_5k_7k', energy_5k_7k(decay_spectrum))
    pool.add('energy_7k_10k', energy_7k_10k(decay_spectrum))
    pool.add('energy_10k_15k', energy_10k_15k(decay_spectrum))

# init feature extractors
w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()
zcr = ZeroCrossingRate()
dish = DistributionShape()
cm = CentralMoments(range=22050)
ds = DistributionShape()

# spectrum energy bands
energy_40_70 = EnergyBand(startCutoffFrequency=40, stopCutoffFrequency=70)
energy_70_110 = EnergyBand(startCutoffFrequency=70, stopCutoffFrequency=110)
energy_130_145 = EnergyBand(startCutoffFrequency=130, stopCutoffFrequency=145)
energy_160_190 = EnergyBand(startCutoffFrequency=160, stopCutoffFrequency=190)
energy_300_400 = EnergyBand(startCutoffFrequency=300, stopCutoffFrequency=400)
energy_5k_7k = EnergyBand(startCutoffFrequency=5000, stopCutoffFrequency=7000)
energy_7k_10k = EnergyBand(startCutoffFrequency=7000, stopCutoffFrequency=10000)
energy_10k_15k = EnergyBand(startCutoffFrequency=10000, stopCutoffFrequency=15000)

# Training phase
training_pools = {}
for classifier in config.CLASSIFIERS:
    pool = essentia.Pool()
    for sample in utils.sampleGenerator(basedir=config.TRAINING_DIR + classifier):
        computeFeatures(pool, sample)
    training_pools[classifier] = PoolAggregator(defaultStats=['mean'])(pool)
    

# Test phase
results = []
for sample in utils.sampleGenerator(basedir=config.SAMPLES_DIR):
    pool = essentia.Pool()
    computeFeatures(pool, sample)
    clf = neighbors.KNeighborsClassifier(15)
    training = []
    labels = []
    for key in training_pools.keys():
        labels.append(key)
        training.append(poolToNumpy(training_pools[key]))
    clf.fit(training, labels)
    test = poolToNumpy(pool)
    results.append(clf.predict(test))

print results
