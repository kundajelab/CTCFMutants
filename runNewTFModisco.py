import sys
import argparse
import h5py
import numpy as np
import pybedtools as bt
import modisco
import modisco.backend
import modisco.nearest_neighbors
import modisco.affinitymat
import modisco.tfmodisco_workflow.seqlets_to_patterns
import modisco.tfmodisco_workflow.workflow
import modisco.aggregator
import modisco.cluster
import modisco.core
import modisco.coordproducers
import modisco.metaclusterers
import modisco.util
from collections import OrderedDict
from sequenceOperationsModiscoPrep import makeSummitPlusMinus, \
    makePositiveSequenceInputArraysFromNarrowPeaks, \
    makePositiveSequenceInputArraysFromFasta, \
    makeSequenceInputArraysFromDifferentialPeaks

"""
This script runs TF-MoDISco on deepLIFT scores for a list of peaks and a corresponding genome.
The predictions are made with a model trained using keras version 0.3.2 or 1.2.2.
To run: python runNewTFModisco [options]
"""

def parseArgument():
    # Parse the input
    parser=argparse.ArgumentParser(description=\
        "Run the new TF-Modisco on deepLIFT scores")
    parser.add_argument("--sequenceLength", type=int, required=False, \
        default=1000, \
        help='Length of the sequence that was given to the model')
    parser.add_argument("--maxPeakLength", type=int, required=False, \
        default=None, \
        help='Maximum length of peaks that will be inputted into the model')
    parser.add_argument("--peakInfoFileName", required=True,\
        help='narrowPeak file with the optimal peaks or name of file with list of summits')
    parser.add_argument("--DESeq2OutputFileName", required=False,\
        help='name of the file with the output from DESeq2, use only if doing differential peaks')
    parser.add_argument("--genomeFileName", required=True,\
        help='name of file with the genome sequence')
    parser.add_argument("--chroms", required=False, action='append', \
        default = ["chr8", "chr9"], \
        help='chromosomes for which deepLIFT scores will be generated')
    parser.add_argument("--differentialPeaks", action='store_true', required=False,\
        help='Data is from a differential peak tasks, so load files for DESeq2')
    parser.add_argument("--negSet", action='store_true', required=False,\
        help='Data is from a differential peak tasks and deepLIFT should be run on the negative set')
    parser.add_argument("--deepLiftScoresFileName", required=True,\
        help='hdf5 file with deepLIFT scores')
    parser.add_argument("--TFModiscoResultsFileName", required=True,\
        help='hdf5 file with TF Modisco results')
    parser.add_argument("--codePath", required=True, \
	help='Path to code that will be used')
    options = parser.parse_args()
    return options

def loadSequenceData(options):
    # Load the sequence data for TF-MoDISco
    # One-hot-encode the true positives
    peakInfoFileNameElements = options.peakInfoFileName.split(".")
    peakInfoFileNamePrefix = ".".join(peakInfoFileNameElements[0:-2])
    # Load the input data
    trueLabelIndices = np.empty((1,1))
    trueLabelRegions = bt.BedTool()
    X = None # Will have numpy arrays of the one-hot-encoded sequences
    Y = None # Will have numpy arrays of the labels
    if (not options.differentialPeaks):
        # The peaks are not differential peaks
        assert (not options.negSet)
        if ((not options.peakInfoFileName.endswith("fa")) and \
            (not options.peakInfoFileName.endswith("fasta"))) and \
            (not options.peakInfoFileName.endswith("fna")):
            # The input is a narrowPeak file
            optimalBedFileName = peakInfoFileNamePrefix + ".bed"
            summitPlusMinus =\
                makeSummitPlusMinus(optimalBedFileName, createOptimalBed=False, \
                    dataShape=(1,4,options.sequenceLength), bedFilegzip=False, \
                    chroms=options.chroms, maxPeakLength=options.maxPeakLength)
            X, Y =\
                makePositiveSequenceInputArraysFromNarrowPeaks(options.peakInfoFileName, \
                    options.genomeFileName, createOptimalBed=False, \
                    dataShape=(1,4,options.sequenceLength), \
                    chroms=options.chroms, multiMode=False, \
                    maxPeakLength=options.maxPeakLength)
        else:
            # The input is a fasta file
            X, Y = makePositiveSequenceInputArraysFromFasta(options.peakInfoFileName, \
                dataShape=(1,4,options.sequenceLength))
    else:
        # The peaks are differential peaks, so only the summits have been included
        summitPlusMinus =\
            makeSummitPlusMinus(options.peakInfoFileName, createOptimalBed=False, \
                dataShape=(1,4,options.sequenceLength), summitCol=1, \
                startCol=None)
        X, Y, _, _, _, indices =\
            makeSequenceInputArraysFromDifferentialPeaks(options.DESeq2OutputFileName, \
                options.genomeFileName, options.peakInfoFileName, \
                (1,4,options.sequenceLength), createOptimalBed=False, \
                backgroundSummitPresent=False, backgroundSummitOnly=True, \
                createModelDir=False, chroms=options.chroms, bigWigFileNames=[], \
                multiMode=False, streamData=False, dataFileName="", \
                RC=False)
    data = None
    data = X[Y == 0,:,:,:] if options.negSet else X[Y == 1,:,:,:]
    dataReshape = np.zeros((data.shape[0], data.shape[3], data.shape[2]))
    dataReshape = np.zeros((data.shape[0], data.shape[3], data.shape[2]))
    for i in range(data.shape[0]):
        # Iterate through the sequences and re-format them to be in the format needed for the new TF-Modisco
        for j in range(data.shape[3]):
            # Iterate through the sequences and re-format the channels
            for k in range(data.shape[2]):
                # Iterate through the channels and enter the re-formatted entries
                dataReshape[i,j,k] = data[i,0,k,j]
    return dataReshape

def loadScoreData(options):
    # Prepare the deepLIFT scores for TF-Modisco
    task_to_scores = OrderedDict()
    task_to_hyp_scores = OrderedDict()
    f = h5py.File(options.deepLiftScoresFileName, "r")
    tasks = f["contrib_scores"].keys()
    for task in tasks:
        # Iterate through the tasks and put the importance and hypothetical importance scores for each task into the dictionaries
        task_to_scores[task] = [np.array(x) for x in f['contrib_scores'][task][:]]
        task_to_hyp_scores[task] =\
            [np.array(x) for x in f['hyp_contrib_scores'][task][:]]
    return task_to_scores, task_to_hyp_scores

def runNewTFModisco(options):
    # Run the new TF-Modisco on deepLIFT scores
    dataReshape = loadSequenceData(options)
    task_to_scores, task_to_hyp_scores = loadScoreData(options)
    # Run TF-Modisco
    reload(modisco)
    reload(modisco.backend.tensorflow_backend)
    reload(modisco.backend)
    reload(modisco.nearest_neighbors)
    reload(modisco.affinitymat.core)
    reload(modisco.affinitymat.transformers)
    reload(modisco.tfmodisco_workflow.seqlets_to_patterns)
    reload(modisco.tfmodisco_workflow.workflow)
    reload(modisco.aggregator)
    reload(modisco.cluster.core)
    reload(modisco.cluster.phenograph.core)
    reload(modisco.cluster.phenograph.cluster)
    reload(modisco.core)
    reload(modisco.coordproducers)
    reload(modisco.metaclusterers)
    print("Running TF-Modisco")
    tfmodisco_results =\
        modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(sliding_window_size=21, \
            flank_size=10, target_seqlet_fdr=0.2, \
	    seqlets_to_patterns_factory =\
	        modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(trim_to_window_size=20, \
		    initial_flank_to_add=5, kmer_len=8, num_gaps=1, \
                    num_mismatches=0, final_min_cluster_size=20))\
	    (task_names = ["task0"], contrib_scores = task_to_scores, \
                hypothetical_contribs = task_to_hyp_scores, one_hot = dataReshape)
    # Save the results
    print("Saving TF-Modisco results")
    reload(modisco.util)
    grp = h5py.File(options.TFModiscoResultsFileName)
    tfmodisco_results.save_hdf5(grp)

if __name__ == "__main__":
        options = parseArgument()
        runNewTFModisco(options)
