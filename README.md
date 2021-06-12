# CTCFMutants
This repository contains the wrapper for TF-MoDISco (Shrikumar et al., "TF-MoDISco v0.4.4.2-alpha: Technical Note" arXiv, 2018) that was used for obtaining TF-MoDISco motifs from deep convolutional neural networks trained to predict whether a CTCF ChIP-seq peak would have significantly lower in a dataset from CTCF with a mutated zinc finger as well as ipython notebooks for visualizing the TF-MoDISco results from those neural networks.
## Code:
* runNewTFModisco.py: TF-MoDISco wrapper
* ipython notebooks: code for visualizing results for each neural network, where the zinc finger number in the notebook name indicates the zinc finger mutant corresponding to the model; require data from http://mitra.stanford.edu/kundaje/imk1/CTCFMutantsProject/TFMoDIScoMotifs/
## Dependencies:
* python 2.7.15
* numpy 1.14.3
* matplotlib 2.2.3
* h5py 2.6.0
* seaborn 0.9.0
* modisco 0.5.1.1 or 0.5.5.6
* pybedtools 0.7.8
* biopython 1.68
* cython 0.29.12
