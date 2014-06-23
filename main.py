import essentia
import utils
import sys
import os.path
import pickle
import config
import demo
import numpy as np
from essentia.standard import *
from sklearn import neighbors
from sklearn.decomposition import PCA
from pylab import *

# init feature extractors & algorithms
w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()
zcr = ZeroCrossingRate()
dish = DistributionShape()
cm = CentralMoments(range=22050)
ds = DistributionShape()
pca = PCA(n_components=2)
knnclf = neighbors.KNeighborsClassifier(weights='distance')
energy_40_70 = EnergyBand(startCutoffFrequency=40, 
                          stopCutoffFrequency=70)
energy_70_110 = EnergyBand(startCutoffFrequency=70, 
                           stopCutoffFrequency=110)
energy_130_145 = EnergyBand(startCutoffFrequency=130, 
                            stopCutoffFrequency=145)
energy_160_190 = EnergyBand(startCutoffFrequency=160, 
                            stopCutoffFrequency=190)
energy_300_400 = EnergyBand(startCutoffFrequency=300, 
                            stopCutoffFrequency=400)
energy_5k_7k = EnergyBand(startCutoffFrequency=5000, 
                          stopCutoffFrequency=7000)
energy_7k_10k = EnergyBand(startCutoffFrequency=7000, 
                           stopCutoffFrequency=10000)
energy_10k_15k = EnergyBand(startCutoffFrequency=10000, 
                            stopCutoffFrequency=15000)

# Convert pool to pickleable list
def poolToPickle(pool):
    result = []
    for key in pool.descriptorNames():
        result.append((key, pool[key]))
    return result

# Convert pool to a PCA reduced vector for analysis
def poolToNumpy(pool):
    result = []
    for key in pool.descriptorNames():
        result.append(pool[key])
    pca.fit(np.reshape(np.array(result).flatten(), (-1, 2)))
    return pca.explained_variance_ratio_

# Convert a pickled list to a pool
def pickleToPool(nparr):
    pool = essentia.Pool()
    for tup in nparr:
        pool.add(tup[0], tup[1])
    return pool
    
# Data we need to compile
# -----------------------
# ZCR                       - Scalar - Attack + Decay segments
# Kurtosis                  - Scalar - Decay segment
# Skewness                  - Scalar - Decay segment
# Centroid                  - Scalar - Decay segment
# Energy Descriptors (FFT)  - Vector - Attack + Decay segments  - m bands
# MFCC                      - Vector - Attack + Decay segments  - n bands
# ----------------------------------------------------------------
# So we have 5 scalar features and m + n vector features
def computeFeatures(pool, sample):
    # Split into attack/decay segments
    attack, decay = utils.extractEnvelopeSegments(sample)

    # Store decay spectrum
    decay_spectrum = spectrum(decay)

    # ZCR
    if len(attack) > 0:
        pool.add('zcr.attack', zcr(attack))
    else:
        pool.add('zcr.attack', 0)
    pool.add('zcr.decay', zcr(decay))

    # Centroid (the first of five extracted moments)
    spectralCentroid = cm(decay_spectrum)
    pool.add('centroid', spectralCentroid[0])

    # Skewness/kurtosis
    _, skewness, kurtosis = ds(spectralCentroid)
    pool.add('skewness', skewness)
    pool.add('kurtosis', kurtosis)

    # MFCC ignoring bands
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


# Training phase
training_pools = {}
for classifier in config.CLASSIFIERS:
    # Load from file if we have training data already
    if os.path.isfile(classifier + '.sig'):
        fd = open(classifier + '.sig', 'r+')
        training_pools[classifier] = pickleToPool(pickle.load(fd))
        fd.close()
        continue

    # Compute features for each training sample
    pool = essentia.Pool()
    basedir = config.TRAINING_DIR + classifier
    for sample, _ in utils.sampleGenerator(basedir=basedir):
        computeFeatures(pool, sample)

    # Aggregate training data to mean for each feature
    training_pools[classifier] = PoolAggregator(defaultStats=['mean'])(pool)

    # Save to disk
    fd = open(classifier + '.sig', 'w+')
    pickle.dump(poolToPickle(training_pools[classifier]), fd)
    fd.close()

# Test phase
results = []
for sample, samplePath in utils.sampleGenerator(basedir=config.SAMPLES_DIR):
    # Compute features for each test sample
    pool = essentia.Pool()
    computeFeatures(pool, sample)

    # PCA-reduce training data and associate with a label  
    training = []
    labels = []
    for key in training_pools.keys():
        labels.append(key)
        training.append(poolToNumpy(training_pools[key]))
    knnclf.fit(training, labels)

    # Predict the label for this sample based on the PCA-reduced test data
    test = poolToNumpy(pool)
    results.append((knnclf.predict(test), samplePath))

# Display results
i = 0
for result in results:
    if result[0] != demo.expected_results[result[1].split('/')[-1]]:
        i += 1
        print str(result[0]) + ' incorrect for ' + str(result[1].split('/')[-1])

print str(len(results) - i) + "/" + str(len(results)) + " correct predictions"
