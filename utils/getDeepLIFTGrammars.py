import sys
import argparse
import numpy as np
import os
import time
import pickle
from argparse import Namespace
#Import some general util stuff
scriptsDir = os.environ.get("UTIL_SCRIPTS_DIR")
if (scriptsDir is None):
    raise Exception("Please set environment variable UTIL_SCRIPTS_DIR to point to av_scripts")
sys.path.insert(0,scriptsDir)
#import latest deepLIFT stuff
scriptsDir = os.environ.get("DEEPLIFT_DIR")
if (scriptsDir is None):
    raise Exception("Please set environment variable DEEPLIFT_DIR to point to the deeplift code (WITHIN the deeplift repo)")
sys.path.insert(0,scriptsDir)
#Make sure the directory is set to import the lab's version of keras
scriptsDir = os.environ.get("KERAS_DIR")
if (scriptsDir is None):
    raise Exception("Please set environment variable KERAS_DIR")
sys.path.insert(0,scriptsDir)
#import old deepLIFT stuff
scriptsDir = os.environ.get("ENHANCER_SCRIPTS_DIR")
if (scriptsDir is None):
    raise Exception("Please set environment variable ENHANCER_SCRIPTS_DIR to point to enhancer_prediction_code")
sys.path.insert(0,scriptsDir+"/featureSelector/deepLIFFT/")
print(scriptsDir)
import deepLIFTutils
sys.path.insert(0,scriptsDir+"/featureSelector/deepLIFFT/kerasBasedBackprop")

def parseArgument():
	# Parse the input
	parser=argparse.ArgumentParser(description=\
			"Get the grammars from the data")
	parser.add_argument("--modelWeightsFileName", required=True,\
			help='hdf5 file with model weights')
	parser.add_argument("--modelArchitectureFileName", required=False,\
			default="/srv/scratch/imk1/TFBindingPredictionProject/EncodeCTCFData/KerasModels/hybrid_CTCF_Helas3_vsDNase_1000bp_conv3LowFiltTopHap_architecture.json", \
			help='json file with model architecture')
	parser.add_argument("--sequenceLength", type=int, required=False, default=1000, \
			help='Length of the sequence that was given to the model')
	parser.add_argument("--maxPeakLength", type=int, required=False, default=None, \
			help='Maximum length of peaks that will be inputted into the model')
	parser.add_argument("--peakInfoFileName", required=True,\
			help='narrowPeak file with the optimal peaks, name of file with list of summits, or name of optimal peak fasta file')
	parser.add_argument("--inputFasta", action='store_true', required=False,\
                        help='The input file is a fasta file')
	parser.add_argument("--createOptimalBed", action='store_true', required=False,\
			help='Create the bed file for the optimal peaks')
	parser.add_argument("--DESeq2OutputFileName", required=False,\
			help='name of the file with the output from DESeq2, use only if doing differential peaks')
	parser.add_argument("--genomeFileName", required=False,\
			default = "/mnt/data/annotations/by_organism/human/hg20.GRCh38/GRCh38.genome.fa", help='name of file with the genome sequence')	
	parser.add_argument("--chroms", required=False, action='append', \
			default = ["chr8", "chr9"], \
			help='chromosomes for which deepLIFT scores will be generated')
	parser.add_argument("--differentialPeaks", action='store_true', required=False,\
			help='Data is from a differential peak tasks, so load files for DESeq2')
	parser.add_argument("--keepFastas", action='store_true', required=False,\
			help='Do not remove the fasta files after they have been created')
	parser.add_argument("--negSet", action='store_true', required=False,\
			help='Data is from a differential peak tasks and deepLIFT should be run on the negative set')
	parser.add_argument("--windowSize", type=int, required=False, default=15, \
			help='Window size and flank size')
	parser.add_argument("--exclusionSize", type=int, required=False, default=15, \
			help='Exclusion size for grammars')
	parser.add_argument("--numGrammars", type=int, required=False, default=8000, \
			help='Number of grammars to use')
	parser.add_argument("--clusteringLayerIndex", type=int, required=False, default=7, \
			help='Index of layer that will be used for clustering; use 7 for a 3-layer network w/o intermediate pooling and 1 for a network w/intermediate pooling')
	parser.add_argument("--lastLayerIndex", type=int, required=False, default=-1, \
			help='Index of the last layer, should be -1 for Keras 0.3.2 and -2 for Keras 1.1.1')
	parser.add_argument("--kerasVersion", type=int, required=False, default=0, \
			help='Version of Keras')
	parser.add_argument("--grammarsFileName", required=True,\
			help='Name of json file where the grammars will be saved')
	parser.add_argument("--grammarCorrMatFileName", required=False,\
			help='Name of file where the grammar correlation matrix will be saved')
	parser.add_argument("--filterGrammars", action='store_true', required=False,\
			help='Filter the grammars to keep only those with a maximum signal that is above the median signal')
	parser.add_argument("--filterFraction", type=float, required=False, default=0.75, \
			help='Fraction of grammars that will be kept during filtering')
	parser.add_argument("--grammarsOnly", action='store_true', required=False,\
			help='Only get grammars (do not get the correlation matrices)')
	parser.add_argument("--numThreads", type=int, required=False, default=1, help='Number of threads to use')
	parser.add_argument("--xcorBatchSize", type=int, required=False, default=10, help='Batch size for xcor')
	parser.add_argument("--funcParamsSize", type=int, required=False, default=1000000, help='Function parameter size for making the cross-correlation matrix')
	parser.add_argument("--codePath", required=False, default="/srv/scratch/imk1/TFBindingPredictionProject/src/", help='Path to code that will be used')
	options = parser.parse_args()
	return options
	
def getTruePositiveIndices(XTrain, YTrain, model, taskNum, lastLayerIdx=-1, kerasVersion=0):
	# Get the indices of the true positives in the dataset
	# Get all the outputs on the data set
	outputs_singleNeuron = model.predict_proba(XTrain)[:,taskNum];
	trueLabels_singleNeuron = YTrain.tolist()
	if len(YTrain.shape) > 1:
		# There are multiple tasks
		trueLabels_singleNeuron = [x[taskNum] for x in YTrain];
	# Compute the true positives because we restrict our attention to them
	truePositiveIndices = deepLIFTutils.getTruePositiveIndicesAboveThreshold(
							outputs=outputs_singleNeuron,
							trueLabels=trueLabels_singleNeuron,
							thresholdProb=0.5);
	return truePositiveIndices
	
def getTrueNegativeIndices(XTrain, YTrain, model, taskNum):
	# Get the indices of the true positives in the dataset
	# Get all the outputs on the data set
	outputs = deepLIFTutils.getSequentialModelLayerOutputs(
				model,
				inputDatas=[XTrain],
				layerIdx=-1,
				batchSize=5);
	# Get the specific outputs for the task of interest
	outputs_singleNeuron = [x[taskNum] for x in outputs];
	trueLabels_singleNeuron = YTrain.tolist()
	if len(YTrain.shape) > 1:
		# There are multiple tasks
		trueLabels_singleNeuron = [x[taskNum] for x in YTrain];
	# Compute the true positives because we restrict our attention to them
	trueNegativeIndices = deepLIFTutils.getTrueNegativeIndicesAboveThreshold(
							outputs=outputs_singleNeuron,
							trueLabels=trueLabels_singleNeuron,
							thresholdProb=0.5);
	return trueNegativeIndices
	
def getIndicesOfOutliers(filterContribs_singleNeuron):
	# Get the indices of the outlier neurons
	filterScores = np.sum(np.squeeze(filterContribs_singleNeuron),axis=(-1,0));
	print(filterScores.shape)
	# Get the indices above zero
	indicesOfOutliers = [x[0] for x in enumerate(np.abs(filterScores)) if x[1] > 0]
	return indicesOfOutliers
	
def getKeyChannelFilterContribs(options, X, deeplift_model, taskNum):
	# Get the deepLIFT scores for the filter layer
	conv_layer_deeplift_contribs_func =\
		deeplift_model.get_target_contribs_func(find_scores_layer_idx=options.clusteringLayerIndex) # Conv. layer for clustering
	filterContribs_singleNeuron = np.array(conv_layer_deeplift_contribs_func(task_idx=taskNum,
									input_data_list=[X],
									batch_size=10,
									progress_update=1000))
	print(filterContribs_singleNeuron.shape)
	filterContribs_revComp_singleNeuron = np.array(conv_layer_deeplift_contribs_func(
											task_idx=taskNum,
											input_data_list=[X[:,:,::-1,::-1]],
											batch_size=10,
											progress_update=1000))
	indicesOfOutliers = getIndicesOfOutliers(filterContribs_singleNeuron)
	print("num outliers", len(indicesOfOutliers))
	# Run the scoring function on true positives the full dataset
	keyChannels_filterContribs = filterContribs_singleNeuron[:, indicesOfOutliers]
	keyChannels_filterContribs_revComp = filterContribs_revComp_singleNeuron[:, indicesOfOutliers]
	return [keyChannels_filterContribs, keyChannels_filterContribs_revComp]
	
def getGrammarsCustomSettings(options, X, model, deeplift_model, target_multipliers_func, target_contribs_func):
	# Get the grammars from the data
	# Obtain the deepLIFT scores for the training data
	contribs_datas_train = [np.array(target_contribs_func(task_idx=0, input_data_list=[X], batch_size=1000, progress_update=5000))]
	# The segment identifier determines how a grammar ("segment") is extracted given the critical subset
	# (The segments will later be cross-correlated with each other to compute a confusion matrix, so it is preferable to keep them short)
	# The algorithm used for the FixedWindowAroundPeaks identifier is as follows:
	#   Compute sums of the deepLIFT contributions in sliding window of size slidingWindowForMaxSize
	#   Find peaks (points whose sliding window sums are larger than their neighbours; for plateaus, take the middle)
	#   Filter out peaks which are not at least ratioToTopPeakToInclude of the tallest peak
	#   For each peak in order of highest peak first:
	#      Add (peakStart-flankToExpandAroundPeakSize, peakStart+slidingWindowForMaxSize+flankToExpandAroundPeakSize) to your list of identified segments
	#      Filter out any peaks that are within excludePeaksWithinWindow of this peak to your list
	#   Loop until there are no more candidate peaks left or the total number of segments identified is maxSegments
	import criticalSubsetIdentification as csi
	segmentIdentifier =\
		csi.FixedWindowAroundPeaks(slidingWindowForMaxSize=options.windowSize, flankToExpandAroundPeakSize=options.windowSize, \
			excludePeaksWithinWindow=options.exclusionSize, ratioToTopPeakToInclude=0.5, maxSegments=5);
	# Returns an array of crticalSubsetIdentification.Grammalr objects and an array of the indices (corresponding to indicesToGetGrammarOn) that each grammar came from
	# (If you pick a segment identifier other than FullSegment, you can have multiple grammars per sequence)
	print(contribs_datas_train[0].shape)
	grammars, grammarIndices =\
		csi.getSeqlets(rawDeepLIFTContribs=contribs_datas_train[0], indicesToGetSeqletsOn=None, outputsBeforeActivation=None, activation=None, \
			thresholdProb=1.0, segmentIdentifier=segmentIdentifier, numThreads=options.numThreads, secondsBetweenUpdates=120);
	print(len(grammarIndices)) # Get a sense of the total number of grammars
	print(grammarIndices[0:10]) # Look at the indices of the first ten grammars to get a sense of which grammars are from which examples
	if options.filterGrammars:
		# Filter the grammars to keep only those with a maximum signal that is above the median signal [based on code by Johnny Israeli]
		grammars_sorted = sorted(grammars, key= lambda x: x.normedCoreDeepLIFTtrack.max(), reverse=True)
		grammars = grammars_sorted[:int(round(options.filterFraction * len(grammars_sorted)))]
		# Augment tracks with score info and underlying sequence info
	# Augment tracks with score info and underlying sequence info
	print("Computing multipliers")
	sequenceMultipliers_singleNeuron = np.squeeze(np.array(target_multipliers_func(task_idx=0, input_data_list=[X], batch_size=10, progress_update=1000)),axis=1)
	[keyChannels_filterContribs, keyChannels_filterContribs_revComp] = getKeyChannelFilterContribs(options, X, deeplift_model, 0)
	import deeplift.util as deeplift_util
	convLayer_effectiveWidth, convLayer_effectiveStride =\
		deeplift_util.get_lengthwise_effective_width_and_stride(deeplift_model.get_layers()[1:(options.clusteringLayerIndex + 1)])
	print("effective width and stride:", convLayer_effectiveWidth, convLayer_effectiveStride)
	for dataToAugmentWith,name,pseudocount,fullRevCompDataArr,revCompFunc,effectiveWidth,effectiveStride,layerFromAbove \
		in [(np.squeeze(X, axis=1), "sequence", 0.25, None, csi.dnaRevCompFunc, 1, 1, False), \
			(sequenceMultipliers_singleNeuron, "sequence_multipliers", 0.0, None, csi.dnaRevCompFunc, 1, 1, False), \
			(np.squeeze(keyChannels_filterContribs, axis=2), "filter_deeplift", 0.0, np.squeeze(keyChannels_filterContribs_revComp, axis=2), None, \
			convLayer_effectiveWidth, convLayer_effectiveStride, True)]:
		csi.augmentSeqletsWithData(grammars, fullDataArr=dataToAugmentWith, keyName=name, pseudocount=pseudocount, fullRevCompDataArr=fullRevCompDataArr, \
			revCompFunc=revCompFunc, indicesToSubset=None, effectiveStride=effectiveStride, effectiveWidth=effectiveWidth, layerFromAbove=layerFromAbove, \
			fillValue=0)
	contribsForSeqlets = [np.sum(seqlet.summedCoreDeepLIFTtrack) for seqlet in grammars]; # Sort them by highest contributing seqlets
	sortOrder = [x[0] for x in sorted(enumerate(contribsForSeqlets), key=lambda x: -x[1])]; 
	grammarsSorted = [grammars[i] for i in sortOrder];
	grammarsSubset = grammarsSorted
	if options.numGrammars > len(grammarsSorted):
		# Save only a subset of the grammars
		grammarsSubset=grammarsSorted[:options.numGrammars]
	csi.Grammar.saveListOfGrammarsToJson(options.grammarsFileName, grammarsSubset)
	return grammarsSubset
	
def getCorrelationMatrixCustomSettings(options, grammarsSubset):
	# Get the pairwise correlation matrix between a subset of the grammars
	accountForRevComp=True
	# subtracksToInclude represents the set of subtracks to do the
	# cross correlation based on.
	subtracksToInclude=["filter_deeplift"]
	import criticalSubsetIdentification as csi
	grammarsCorrMat =\
		csi.getCorrelationMatrix(grammarsSubset, subtracksToInclude=subtracksToInclude, accountForRevComp=accountForRevComp, numThreads=options.numThreads, \
			secondsBetweenUpdates=120, xcorBatchSize=options.xcorBatchSize, funcParamsSize=options.funcParamsSize)
	np.savetxt(options.grammarCorrMatFileName, grammarsCorrMat, delimiter='\t')

def getDeepLIFTGrammars(options):
	# Get deepLIFT scores, their grammars, and the correlations between the grammars
	sys.path.insert(0,options.codePath)
	from sequenceOperations import makePositiveSequenceInputArraysFromNarrowPeaks, makeSequenceInputArraysFromDifferentialPeaks, \
		makePositiveSequenceInputArraysFromFasta
	print("Loading model!")
	from keras.models import model_from_json
	model = model_from_json(open(options.modelArchitectureFileName).read())
	model.load_weights(options.modelWeightsFileName)
	print(model.layers)
	# Load the input data.
	trueLabelIndices = np.empty((1,1))
	X = np.empty((1,1))
	Y = np.empty((1,1))
	taskNum = 0
	if (not options.differentialPeaks):
		# The peaks are not differential peaks
		assert (not options.negSet)
		if options.inputFasta:
			# The input file is a fasta file
			X, Y = makePositiveSequenceInputArraysFromFasta(options.peakInfoFileName, dataShape=(1,4,options.sequenceLength))
		else:
			# The input file is a narrowPeak file or a list of summits
			X, Y =\
				makePositiveSequenceInputArraysFromNarrowPeaks(options.peakInfoFileName, options.genomeFileName, \
				createOptimalBed=options.createOptimalBed, dataShape=(1,4,options.sequenceLength), chroms=options.chroms, multiMode=False, \
				maxPeakLength=options.maxPeakLength)
		#Load the input data
		trueLabelIndices = getTruePositiveIndices(X, Y, model, taskNum, lastLayerIdx=options.lastLayerIndex, kerasVersion=options.kerasVersion)
	else:
		# The peaks are differential peaks, so only the summits have been included
		X, Y, _, _, _, indices =\
			makeSequenceInputArraysFromDifferentialPeaks(options.DESeq2OutputFileName, options.genomeFileName, options.peakInfoFileName, \
				(1,4,options.sequenceLength), createOptimalBed=False, backgroundSummitPresent=False, backgroundSummitOnly=True, createModelDir=False, \
				chroms=options.chroms, bigWigFileNames=[], multiMode=False, streamData=False, dataFileName="", RC=False, removeFastas=(not options.keepFastas))
		if (not options.negSet):
			# Run deepLIFT on the positives
			trueLabelIndices = getTruePositiveIndices(X, Y, model, taskNum, lastLayerIdx=options.lastLayerIndex)
		else:
			# Run deepLIFT on the negatives
			trueLabelIndices = getTrueNegativeIndices(X, Y, model, taskNum)
	data = X[trueLabelIndices]
	print("Last true label index: " + str(trueLabelIndices[-1]))
	# Covnert the model
	import deeplift.conversion.keras_conversion as kc
	deeplift_model = kc.convert_sequential_model(model)
	print(deeplift_model.get_layers())
	# Compile the functions to compute the contributions and multipliers - the multipliers are analogous to the gradients
	target_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)
	target_multipliers_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0)
	if options.negSet:
		# Multiply the deepLIFT scores by -1 because using the negative set
		target_contribs_func = 0 - target_contribs_func
		target_multipliers_func = 0 - target_multipliers_func
	# The initial stage is to compute a distance matrix between the grammars to identify clusters
	grammarsSubset = getGrammarsCustomSettings(options, data, model, deeplift_model, target_multipliers_func, target_contribs_func)
	if not (options.grammarsOnly):
		# Compute the pairwise distance matrix between the grammars
		# This is by far the most time-consuming operation, so multithread
		getCorrelationMatrixCustomSettings(options, grammarsSubset)

if __name__ == "__main__":
	options = parseArgument()
	getDeepLIFTGrammars(options)
