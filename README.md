# CTCFMutants
This repository contains the wrapper for TF-MoDISco (Shrikumar et al., "TF-MoDISco v0.4.4.2-alpha: Technical Note," arXiv, 2018) that was used for obtaining TF-MoDISco motifs from deep convolutional neural networks trained to predict whether a CTCF ChIP-seq peak would have significantly lower in a dataset from CTCF with a mutated zinc finger as well as ipython notebooks for visualizing the TF-MoDISco results from those neural networks.  It also contains scripts for analyses involving the TF-MoDISco results.
## Code for General Use:
* runNewTFModisco.py: TF-MoDISco wrapper
* sequenceOperationsModiscoPrep.py: utilities used by runNewTFModisco.py
* ipython notebooks: code for visualizing results for each neural network, where the zinc finger number in the notebook name indicates the zinc finger mutant corresponding to the model; require data from http://mitra.stanford.edu/kundaje/imk1/CTCFMutantsProject/TFMoDIScoMotifs/
* convertFIMOToMotifHitBed.py: converts an output file from FIMO (Grant et al., "FIMO: Scanning for occurrences of a given motif," Bioinformatics, 2011) to a bed file
## Code for Analyses in Kaplow et al. (in evaluationScripts):
* analyzeCTCFsDataPlus.sh: code for analyses involving CTCF-s data (Le et al., "An alternative CTCF isoform antagonizes canonical CTCF occupancy and changes chromatin architecture to promote apoptosis," Nature Communications, 2019)
* analyzeCTCFUpstreamDownstreamNewTFModiscoMotifsAllHitsPlus.sh: code for analyses of mouse activated B cell peaks (Nakahashi et al., "A genome-wide map of CTCF multivalency redefines the CTCF code," Cell Reports, 2013) overlapping the core, upstream, and downstream motifs with no FIMO motif hit cutoff
* analyzeCTCFUpstreamDownstreamNewTFModiscoMotifsHeartAllHits.sh: code for analyses of mouse heart peaks (mouse ENCODE) overlapping the core, upstream, and downstream motifs with no FIMO motif hit cutoff
* analyzeCTCFUpstreamDownstreamNewTFModiscoMotifsHeartqVal.sh: code for analyses of mouse heart peaks overlapping the core, upstream, and downstream motifs with the FIMO motif hit q-value < 0.05 cutoff
* analyzeCTCFUpstreamDownstreamNewTFModiscoMotifsHeart.sh: code for analyses of mouse heart peaks overlapping the core, upstream, and downstream motifs with the default FIMO motif hit cutoff
* analyzeCTCFUpstreamDownstreamNewTFModiscoMotifsLiverAllHits.sh: code for analyses of mouse liver peaks (mouse ENCODE) overlapping the core, upstream, and downstream motifs with no FIMO motif hit cutoff
* analyzeCTCFUpstreamDownstreamNewTFModiscoMotifsLiverqVal.sh: code for analyses of mouse liver peaks overlapping the core, upstream, and downstream motifs with the FIMO motif hit q-value < 0.05 cutoff
* analyzeCTCFUpstreamDownstreamNewTFModiscoMotifsLiver.sh: code for analyses of mouse liver peaks overlapping the core, upstream, and downstream motifs with the default FIMO motif hit cutoff
* analyzeCTCFUpstreamDownstreamNewTFModiscoMotifsPlus.sh: code for analyses of mouse activated B cell peaks overlapping the core, upstream, and downstream motifs with the default FIMO motif hit cutoff
* analyzeCTCFUpstreamDownstreamNewTFModiscoMotifsqValPlus.sh: code for analyses of mouse activated B cell peaks overlapping the core, upstream, and downstream motifs with the FIMO motif hit q-value < 0.05 cutoff
* deseq2Script.r: code for obtaining differential peaks between wild type CTCF ChIP-seq and CTCF ChIP-seq with the zinc finger 1 mutant
* getDeepLiftScoresCTCFMutantsBigWigs.sh: code for obtaining DeepLIFT score bigwig files for each of the wild type CTCF verses mutant CTCF binding prediction models
* getDeepLiftScoresCTCFMutants.sh: code for obtaining DeepLIFT scores for each of the wild type CTCF versus mutant CTCF binding prediction models
* runNewTFModiscoCTCFMutants.sh: code for running TF-MoDISco on DeepLIFT scores for each of the wild type CTCF versus mutant CTCF binding prediction models
## Utilities for Code in evaluationScripts (in utils):
* getBestFIMOBed.py: gets the best motif hit from FIMO in a bed file
* makeViolinPlotsCoreDownstreamCTCFs.py: make violin plots for CTCFs analysis visualizations
* getDeepLIFTScores.py: wrapper for DeepLIFT for models trained using Keras 0.3.2 with the Theano backend
* makeBedGraphFromPositionScores.py: makes a single bedGraph file from a text file with per-position DeepLIFT scores
* makeBedGraphFromPositionScoresPerSequence.py: makes a bedGraph file for each sequence from a text file with per-position DeepLIFT scores
* getDeepLIFTScoresCrossVal.py: wrapper for DeepLIFT for models trained using Keras 0.3.2 with the Theano background that iterates through cross-validation folds
* sequenceOperations.py: utilities for converting DNA sequence files into the numpy files for training deep learning models
* getDeepLIFTGrammars.py: wrapper for earlier version of TF-MoDISco that contains utilities for subsetting sequences based on deep learning model predictions
## Dependencies:
* python 2.7.15 (required for ipython notebooks, evaluationScripts, and utils) or 3.7.1
* numpy 1.14.3 (python 2) or 1.17.0 (python 3)
* matplotlib 2.2.3 (python 2) or 3.0.2 (python 3)
* h5py 2.6.0 (python 2) or 2.10.0 (python 3)
* seaborn 0.9.0 (required for only ipython notebooks)
* modisco 0.5.1.1 (python 2), 0.5.5.6 (python 2), or 0.5.14.1 (python 3)
* pybedtools 0.7.8 (python 2) or 0.8.1 (python 3)
* biopython 1.68 (python 2) or 1.73 (python 3)
* cython 0.29.12 (python 2) or 0.29.13 (python 3)
* meme 4.12.0 (evaluationScripts only)
* R 3.5.1 (evaluationScripts only)
* DESeq2 1.22.2 (evaluationScripts only)
* pybedtools 0.7.8 (evaluationScripts only)
* deeplift 0.5.5-theano (evaluationScripts only)
* keras 0.3.2 (evaluationScripts only)
* bedGraphToBigWig (http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bedGraphToBigWig) (evaluationScripts only)
* Biopython 1.68 (evaluationScripts only)
* cython 0.29.12 (evaluationScripts only)
