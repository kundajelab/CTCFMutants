import sys
import argparse
import numpy as np
import os
from argparse import Namespace
import pybedtools as bt
#Import some general util stuff
scriptsDir = os.environ.get("UTIL_SCRIPTS_DIR")
if (scriptsDir is None):
    raise Exception("Please set environment variable UTIL_SCRIPTS_DIR to point to av_scripts")
sys.path.insert(0,scriptsDir)
import pathSetter
import util
#import old deepLIFT stuff
scriptsDir = os.environ.get("ENHANCER_SCRIPTS_DIR")
if (scriptsDir is None):
    raise Exception("Please set environment variable ENHANCER_SCRIPTS_DIR to point to enhancer_prediction_code")
sys.path.insert(0,scriptsDir+"/featureSelector/deepLIFFT/")
import deepLIFTutils
sys.path.insert(0,scriptsDir+"/featureSelector/deepLIFFT/kerasBasedBackprop")
from makeBedGraphFromPositionScores import makeBedGraphFromPositionScores
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
from keras.models import model_from_json
import deeplift.conversion.keras_conversion as kc

def parseArgument():
	# Parse the input
	parser=argparse.ArgumentParser(description=\
			"Make the files of deepLIFT scores")
	parser.add_argument("--modelWeightsFileNamePrefix", required=True,\
			help='Prefix of hdf5 file with model weights')
	parser.add_argument("--modelArchitectureFileName", required=False,\
			default="/srv/scratch/imk1/TFBindingPredictionProject/EncodeCTCFData/KerasModels/hybrid_CTCF_Helas3_vsDNase_1000bp_conv3LowFiltTopHap_architecture.json", \
			help='json file with model architecture')
	parser.add_argument("--sequenceLength", type=int, required=False, default=1000, \
			help='Length of the sequence that was given to the model')
	parser.add_argument("--numModels", type=int, required=False, default=5, \
			help='Number of cross-validation folds, which is the number of models that will be loaded')			
	parser.add_argument("--optimalPeakFileName", required=True,\
			help='narrowPeak file with the optimal peaks')
	parser.add_argument("--genomeFileName", required=False,\
			default = "/mnt/data/annotations/by_organism/human/hg20.GRCh38/GRCh38.genome.fa", help='name of file with the genome sequence')	
	parser.add_argument("--genomeName", required=False,\
			default = "hg38", help='name of the genome assembly')
	parser.add_argument("--chroms", required=False, action='append', \
			default = ["chr1", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr2", "chr20", "chr21", \
				"chr22", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chrX"], \
			help='chromosomes for which deepLIFT scores will be generated')
	parser.add_argument("--outputFileNamePrefix", required=True,\
			help='Prefix for score and bedgraph output files, should not end with _')
	parser.add_argument("--scoreTrackPrefix", required=True, \
			help='Prefix for bedgraph score tracks')
	parser.add_argument("--maxOnly", action='store_true', required=False,\
			help='Only get maximum scores (do not get the nucleotide-specific scores, reverse complements will not be used)')
	parser.add_argument("--codePath", required=False, default="/srv/scratch/imk1/TFBindingPredictionProject/src/", help='Path to code that will be used')
	options = parser.parse_args()
	return options

def makeDeepLIFTScoreFilesCrossVal(options, contribs_datas, i):
	# Make the files of deepLIFT scores for the current cross-validation iteration
	contribs_dataFileNameA = "_".join([options.outputFileNamePrefix, "deepLIFTScoresA", "fold" + str(i) + ".txt"])
	contribs_dataFileNameC = "_".join([options.outputFileNamePrefix, "deepLIFTScoresC", "fold" + str(i) + ".txt"])
	contribs_dataFileNameG = "_".join([options.outputFileNamePrefix, "deepLIFTScoresG", "fold" + str(i) + ".txt"])
	contribs_dataFileNameT = "_".join([options.outputFileNamePrefix, "deepLIFTScoresT", "fold" + str(i) + ".txt"])
	contribs_dataFileNameMax = "_".join([options.outputFileNamePrefix, "deepLIFTScoresMax", "fold" + str(i) + ".txt"])
	contribs_data_A = np.reshape(contribs_datas[0][:,0,0,:], (contribs_datas[0].shape[0], contribs_datas[0].shape[3]))
	contribs_data_C = np.reshape(contribs_datas[0][:,0,1,:], (contribs_datas[0].shape[0], contribs_datas[0].shape[3]))
	contribs_data_G = np.reshape(contribs_datas[0][:,0,2,:], (contribs_datas[0].shape[0], contribs_datas[0].shape[3]))
	contribs_data_T = np.reshape(contribs_datas[0][:,0,3,:], (contribs_datas[0].shape[0], contribs_datas[0].shape[3]))
	contribs_data_Max = np.reshape(np.max(contribs_datas[0][:,0,:,:], axis = 1), (contribs_datas[0].shape[0], contribs_datas[0].shape[3]))
	if not options.maxOnly:
		# Save the per-nucleotide scores
		np.savetxt(contribs_dataFileNameA, contribs_data_A, fmt='%f', delimiter='\t')
		np.savetxt(contribs_dataFileNameC, contribs_data_C, fmt='%f', delimiter='\t')
		np.savetxt(contribs_dataFileNameG, contribs_data_G, fmt='%f', delimiter='\t')
		np.savetxt(contribs_dataFileNameT, contribs_data_T, fmt='%f', delimiter='\t')
	np.savetxt(contribs_dataFileNameMax, contribs_data_Max, fmt='%f', delimiter='\t')
	
def makeBedGraphFromPositionScoresAllCrossVal(options, summitPlusMinusFileName, i):
	# Make bedgraph files for every nucleotide for every sequence and its reverse complement for the current cross-validation iteration
	if not options.maxOnly:
		# Get the per-nucleotide scores
		optionsA = Namespace(
			regionsFileName = summitPlusMinusFileName, 
			scoresFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresA", "fold" + str(i) + ".txt"]), 
			outputFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresA", "fold" + str(i) + ".bedgraph"]),
			numScoresPerRegion = 1, trackName = "_".join([options.scoreTrackPrefix, "deepLIFTScoresA", "fold" + str(i)]), 
			chromCol = 0, startCol = 1, endCol = 2, excludeRC=False, onlyRC=False)
		makeBedGraphFromPositionScores(optionsA)
		optionsC = Namespace(
			regionsFileName = summitPlusMinusFileName, 
			scoresFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresC", "fold" + str(i) + ".txt"]), 
			outputFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresC", "fold" + str(i) + ".bedgraph"]),
			numScoresPerRegion = 1, trackName = "_".join([options.scoreTrackPrefix, "deepLIFTScoresC", "fold" + str(i)]), 
			chromCol = 0, startCol = 1, endCol = 2, excludeRC=False, onlyRC=False)
		makeBedGraphFromPositionScores(optionsC)
		optionsG = Namespace(
			regionsFileName = summitPlusMinusFileName, 
			scoresFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresG", "fold" + str(i) + ".txt"]), 
			outputFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresG", "fold" + str(i) + ".bedgraph"]),
			numScoresPerRegion = 1, trackName = "_".join([options.scoreTrackPrefix, "deepLIFTScoresG", "fold" + str(i)]), 
			chromCol = 0, startCol = 1, endCol = 2, excludeRC=False, onlyRC=False)
		makeBedGraphFromPositionScores(optionsG)
		optionsT = Namespace(
			regionsFileName = summitPlusMinusFileName, 
			scoresFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresT", "fold" + str(i) + ".txt"]), 
			outputFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresT", "fold" + str(i) + ".bedgraph"]),
			numScoresPerRegion = 1, trackName = "_".join([options.scoreTrackPrefix, "deepLIFTScoresT", "fold" + str(i)]), 
			chromCol = 0, startCol = 1, endCol = 2, excludeRC=False, onlyRC=False)
		makeBedGraphFromPositionScores(optionsT)
	optionsMax = Namespace(
		regionsFileName = summitPlusMinusFileName, 
		scoresFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresMax", "fold" + str(i) + ".txt"]), 
		outputFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresMax", "fold" + str(i) + ".bedgraph"]),
		numScoresPerRegion = 1, trackName = "_".join([options.scoreTrackPrefix, "deepLIFTScoresMax", "fold" + str(i)]), 
		chromCol = 0, startCol = 1, endCol = 2, excludeRC=False, onlyRC=False)
	makeBedGraphFromPositionScores(optionsMax)
	
def sortBedgraphFilesCrossVal(options, i):
	# Sort bedgraph files and keep the top-scoring value for every position
	if not options.maxOnly:
		# Sort the per-nucleotide score files
		os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresA_fold' + str(i) + '.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresA_fold' + str(i) + '_sorted.bedgraph')
		os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresC_fold' + str(i) + '.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresC_fold' + str(i) + '_sorted.bedgraph')
		os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresG_fold' + str(i) + '.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresG_fold' + str(i) + '_sorted.bedgraph')
		os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresT_fold' + str(i) + '.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresT_fold' + str(i) + '_sorted.bedgraph')
	os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresMax_fold' + str(i) + '.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresMax_fold' + str(i) + '_sorted.bedgraph')
	# Remove the original bedgraphs because they take up a lot of space
	if not options.maxOnly:
		# Remove the per-nucleotide score files
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresA_fold' + str(i) + '.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresC_fold' + str(i) + '.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresG_fold' + str(i) + '.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresT_fold' + str(i) + '.bedgraph')
	os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresMax_fold' + str(i) + '.bedgraph')

def bedgraphToBigwig(bedgraph, genome, output):
	# Convert a bedgraph to a bigwig, based on pybedtools.contrib.bigwig.bedgraph_to_bigwig
	genome_file = bt.chromsizes_to_file(bt.chromsizes(genome))
	cmds = [
		'bedGraphToBigWig',
		bedgraph,
		genome_file,
		output]
	os.system(' '.join(cmds))

def convertBedgraphsToBigwigs(options, i):
	# Convert bedgraphs to bigwigs
	if not options.maxOnly:
		# Convert the per-nucleotide score files to bigwigs
		bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresA", "fold" + str(i), "sorted.bedgraph"]), \
			options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresA", "fold" + str(i) + ".bw"]))
		bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresC", "fold" + str(i), "sorted.bedgraph"]), \
			options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresC", "fold" + str(i) + ".bw"]))
		bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresG", "fold" + str(i), "sorted.bedgraph"]), \
			options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresG", "fold" + str(i) + ".bw"]))
		bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresT", "fold" + str(i), "sorted.bedgraph"]), \
			options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresT", "fold" + str(i) + ".bw"]))
	bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresMax", "fold" + str(i), "sorted.bedgraph"]), \
		options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresMax", "fold" + str(i) + ".bw"]))
	# Remove the sorted bedgraphs because they take up a lot of space
	if not options.maxOnly:
		# Remove the per-nucleotide score files
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresA_fold' + str(i) + '_sorted.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresC_fold' + str(i) + '_sorted.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresG_fold' + str(i) + '_sorted.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresT_fold' + str(i) + '_sorted.bedgraph')
	os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresMax_fold' + str(i) + '_sorted.bedgraph')

def getDeepLIFTScoresCrossVal(options):
	# Get the DeepLIFT scores for a set of regions from a model in bedgraph format
	# Do this for each of 5 cross-validation runs
	# DOES NOT DO REVERSE COMPLEMENTS
	sys.path.insert(0,options.codePath + "deeplearning")
	from sequenceOperations import makePositiveSequenceInputArraysFromNarrowPeaks, makeSummitPlusMinus
	from getDeepLIFTGrammars import getTruePositiveIndices
	model = model_from_json(open(options.modelArchitectureFileName).read())
	optimalPeakFileNameElements = options.optimalPeakFileName.split(".")
	optimalPeakFileNamePrefix = ".".join(optimalPeakFileNameElements[0:-2])
	optimalBedFileName = optimalPeakFileNamePrefix + ".bed"
	summitPlusMinus = makeSummitPlusMinus(optimalBedFileName, createOptimalBed=False, dataShape=(1,4,options.sequenceLength), bedFilegzip=False)
	#Load the input data
	X, Y =\
		makePositiveSequenceInputArraysFromNarrowPeaks(options.optimalPeakFileName, options.genomeFileName, \
			createOptimalBed=False, dataShape=(1,4,options.sequenceLength), chroms=options.chroms, multiMode=False)
	taskNum = 0
	for i in range(options.numModels):
		# Iterate through the models and get the deepLIFT scores for each
		model.load_weights(options.modelWeightsFileNamePrefix + "_fold" + str(i) + ".hdf5")
		# Covnert the model
		deeplift_model = kc.convert_sequential_model(model)
		# Compile the functions to compute the contributions and multipliers - the multipliers are analogous to the gradients
		target_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)
		target_multipliers_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0)
		truePositiveIndices = getTruePositiveIndices(X, Y, model, taskNum)
		data = X[truePositiveIndices]
		print("Last true positive index: " + str(truePositiveIndices[-1]))
		truePosRegions = summitPlusMinus.at(truePositiveIndices)
		truePosRegionsFileName = optimalPeakFileNamePrefix + "_summitsPlusMinus500bp_truePosFold" + str(i) + ".bed"
		truePosRegions.saveas(truePosRegionsFileName)
		# Obtain the deepLIFT scores for the data
		contribs_datas = [np.array(target_contribs_func(task_idx=j, input_data_list=[data], batch_size=1000, progress_update=5000))
							for j in [0]]
		print(np.shape(contribs_datas[0]))
		# Save the deepLIFT scores for the new data
		makeDeepLIFTScoreFilesCrossVal(options, contribs_datas, i)
		# Convert the deepLIFT score files into bedgraph files
		makeBedGraphFromPositionScoresAllCrossVal(options, truePosRegionsFileName, i)
		os.remove(truePosRegionsFileName)
		# Sort the bedgraph files and keep the top score for any position with multiple scores
		sortBedgraphFilesCrossVal(options, i)
		convertBedgraphsToBigwigs(options, i)

if __name__ == "__main__":
	options = parseArgument()
	getDeepLIFTScoresCrossVal(options)
