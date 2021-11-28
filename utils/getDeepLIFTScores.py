import sys
import argparse
import numpy as np
import os
from argparse import Namespace
import pybedtools as bt
import h5py
#Import some general util stuff
scriptsDir = os.environ.get("UTIL_SCRIPTS_DIR")
if (scriptsDir is None):
    raise Exception("Please set environment variable UTIL_SCRIPTS_DIR to point to av_scripts")
sys.path.insert(0,scriptsDir)
#import old deepLIFT stuff
scriptsDir = os.environ.get("ENHANCER_SCRIPTS_DIR")
if (scriptsDir is None):
    raise Exception("Please set environment variable ENHANCER_SCRIPTS_DIR to point to enhancer_prediction_code")
sys.path.insert(0,scriptsDir+"/featureSelector/deepLIFFT/")
sys.path.insert(0,scriptsDir+"/featureSelector/deepLIFFT/kerasBasedBackprop")
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

def parseArgument():
	# Parse the input
	parser=argparse.ArgumentParser(description=\
			"Make the files of deepLIFT scores")
	parser.add_argument("--modelWeightsFileName", required=True,\
			help='hdf5 file with model weights')
	parser.add_argument("--modelArchitectureFileName", required=False,\
			default="/srv/scratch/imk1/TFBindingPredictionProject/MouseMutantData/indivRep_allPeaks-pr1.IDR0.05.mergedPeaks.KerasModels/hybrid_CTCFZF1_actB_vsWT_1000bp_conv3LowFiltTopHap_architecture.json", \
			help='json file with model architecture')
	parser.add_argument("--sequenceLength", type=int, required=False, default=1000, \
			help='Length of the sequence that was given to the model')
	parser.add_argument("--maxPeakLength", type=int, required=False, default=None, \
			help='Maximum length of peaks that will be inputted into the model')
	parser.add_argument("--peakInfoFileName", required=True,\
			help='narrowPeak file with the optimal peaks or name of file with list of summits')
	parser.add_argument("--DESeq2OutputFileName", required=False,\
			help='name of the file with the output from DESeq2, use only if doing differential peaks')
	parser.add_argument("--genomeFileName", required=False,\
			default = "/mnt/data/annotations/by_organism/human/hg20.GRCh38/GRCh38.genome.fa", help='name of file with the genome sequence')	
	parser.add_argument("--genomeName", required=False,\
			default = "hg38", help='name of the genome assembly')
	parser.add_argument("--chroms", required=False, action='append', \
			default = ["chr8", "chr9"], \
			help='chromosomes for which deepLIFT scores will be generated')
	parser.add_argument("--numSequences", type=int, required=False,\
			help='Number of sequences')
	parser.add_argument("--outputFileNamePrefix", required=True,\
			help='Prefix for score and bedgraph output files, should not end with _')
	parser.add_argument("--scoreTrackPrefix", required=True, \
			help='Prefix for bedgraph score tracks')
	parser.add_argument("--differentialPeaks", action='store_true', required=False,\
			help='Data is from a differential peak tasks, so load files for DESeq2')
	parser.add_argument("--negSet", action='store_true', required=False,\
			help='Data is from a differential peak tasks and deepLIFT should be run on the negative set')
	parser.add_argument("--includeIncorrect", action='store_true', required=False,\
                        help='Get the deepLIFT scores for both the correctly and the incorrectly predicted examples')
	parser.add_argument("--getHypotheticalContribs", action='store_true', required=False,\
                        help='Get the hypothetical contributions')
	parser.add_argument("--maxOnly", action='store_true', required=False,\
			help='Only get maximum scores (do not get the nucleotide-specific scores, reverse complements will not be used)')
	parser.add_argument("--separateFilePerSequence", action='store_true', required=False,\
			help='Make a separate file for every sequence')
	parser.add_argument("--hdf5Output", action='store_true', required=False,\
                        help='Make hdf5 files with the deepLIFT scores')
	parser.add_argument("--codePath", required=False, default="/srv/scratch/imk1/TFBindingPredictionProject/src/", help='Path to code that will be used')
	options = parser.parse_args()
	return options

def makeDeepLIFTScoreFiles(options, contribs_datas):
	# Make the files of deepLIFT scores
	contribs_dataFileNameA = options.outputFileNamePrefix+"_deepLIFTScoresA.txt"
	contribs_dataFileNameC = options.outputFileNamePrefix+"_deepLIFTScoresC.txt"
	contribs_dataFileNameG = options.outputFileNamePrefix+"_deepLIFTScoresG.txt"
	contribs_dataFileNameT = options.outputFileNamePrefix+"_deepLIFTScoresT.txt"
	contribs_dataFileNameMax = options.outputFileNamePrefix+"_deepLIFTScoresMax.txt"
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
	
def makeBedGraphFromPositionScoresAll(options, truePosRegionsFileName):
	# Make bedgraph files for every nucleotide for every sequence and its reverse complement
	print ("Making bedgraphs")
	if not options.maxOnly:
		# Get the per-nucleotide scores
		from makeBedGraphFromPositionScores import makeBedGraphFromPositionScores
		optionsA = Namespace(
			regionsFileName = truePosRegionsFileName, 
			scoresFileName = options.outputFileNamePrefix+"_deepLIFTScoresA.txt", 
			outputFileName = options.outputFileNamePrefix+"_deepLIFTScoresA.bedgraph",
			numScoresPerRegion = 1, trackName = options.scoreTrackPrefix+"_deepLIFTScoresA", 
			chromCol = 0, startCol = 1, endCol = 2, excludeRC = False, onlyRC = False)
		makeBedGraphFromPositionScores(optionsA)
		optionsC = Namespace(
			regionsFileName = truePosRegionsFileName, 
			scoresFileName = options.outputFileNamePrefix+"_deepLIFTScoresC.txt", 
			outputFileName = options.outputFileNamePrefix+"_deepLIFTScoresC.bedgraph",
			numScoresPerRegion = 1, trackName = options.scoreTrackPrefix+"_deepLIFTScoresC", 
			chromCol = 0, startCol = 1, endCol = 2, excludeRC = False, onlyRC = False)
		makeBedGraphFromPositionScores(optionsC)
		optionsG = Namespace(
			regionsFileName = truePosRegionsFileName, 
			scoresFileName = options.outputFileNamePrefix+"_deepLIFTScoresG.txt", 
			outputFileName = options.outputFileNamePrefix+"_deepLIFTScoresG.bedgraph",
			numScoresPerRegion = 1, trackName = options.scoreTrackPrefix+"_deepLIFTScoresG", 
			chromCol = 0, startCol = 1, endCol = 2, excludeRC = False, onlyRC = False)
		makeBedGraphFromPositionScores(optionsG)
		optionsT = Namespace(
			regionsFileName = truePosRegionsFileName, 
			scoresFileName = options.outputFileNamePrefix+"_deepLIFTScoresT.txt", 
			outputFileName = options.outputFileNamePrefix+"_deepLIFTScoresT.bedgraph",
			numScoresPerRegion = 1, trackName = options.scoreTrackPrefix+"_deepLIFTScoresT", 
			chromCol = 0, startCol = 1, endCol = 2, excludeRC = False, onlyRC = False)
		makeBedGraphFromPositionScores(optionsT)
	optionsMax = Namespace(
		regionsFileName = truePosRegionsFileName, 
		scoresFileName = options.outputFileNamePrefix+"_deepLIFTScoresMax.txt", 
		outputFileName = options.outputFileNamePrefix+"_deepLIFTScoresMax.bedgraph",
		numScoresPerRegion = 1, trackName = options.scoreTrackPrefix+"_deepLIFTScoresMax", 
		chromCol = 0, startCol = 1, endCol = 2, excludeRC = False, onlyRC = False)
	makeBedGraphFromPositionScores(optionsMax)
	
def makeBedGraphFromPositionScoresPerSequenceAll(options, truePosRegionsFileName):
	# Make a separate bedgraph file for every sequence
	print ("Getting per-sequence position scores!")
	from makeBedGraphFromPositionScoresPerSequence import makeBedGraphFromPositionScoresPerSequence
	optionsMax = Namespace(
		regionsFileName = truePosRegionsFileName, 
		scoresFileName = options.outputFileNamePrefix+"_deepLIFTScoresMax.txt", 
		outputFileNamePrefix = options.outputFileNamePrefix+"_deepLIFTScoresMax",
		numScoresPerRegion = 2, trackName = options.scoreTrackPrefix+"_deepLIFTScoresMax", 
		chromCol = 0, startCol = 1, endCol = 2, excludeRC = True, onlyRC = False)
	makeBedGraphFromPositionScoresPerSequence(optionsMax)
	
def sortBedgraphFiles(options):
	# Sort bedgraph files and keep the top-scoring value for every position
	if not options.maxOnly:
		# Sort the per-nucleotide score files
		os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresA.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresA_sorted.bedgraph')
		os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresC.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresC_sorted.bedgraph')
		os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresG.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresG_sorted.bedgraph')
		os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresT.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresT_sorted.bedgraph')
	os.system('(head -n 1; sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < '+options.outputFileNamePrefix+'_deepLIFTScoresMax.bedgraph | (head -n 1; sort -u -k1,1 -k2,2n -k3,3n) > '+options.outputFileNamePrefix+'_deepLIFTScoresMax_sorted.bedgraph')
	# Remove the original bedgraphs because they take up a lot of space
	if not options.maxOnly:
		# Remove the per-nucleotide score files
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresA.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresC.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresG.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresT.bedgraph')
	os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresMax.bedgraph')
	
def convertBedgraphsToBigwigs(options):
	# Convert bedgraphs to bigwigs
	print("Converting files to bigwigs")
	from getDeepLIFTScoresCrossVal import bedgraphToBigwig
	if not options.maxOnly:
		# Convert the per-nucleotide score files to bigwigs
		bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresA", "sorted.bedgraph"]), \
			options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresA.bw"]))
		bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresC", "sorted.bedgraph"]), \
			options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresC.bw"]))
		bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresG", "sorted.bedgraph"]), \
			options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresG.bw"]))
		bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresT", "sorted.bedgraph"]), \
			options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresT.bw"]))
	bedgraphToBigwig("_".join([options.outputFileNamePrefix, "deepLIFTScoresMax", "sorted.bedgraph"]), \
		options.genomeName, "_".join([options.outputFileNamePrefix, "deepLIFTScoresMax.bw"]))
	# Remove the sorted bedgraphs because they take up a lot of space
	if not options.maxOnly:
		# Remove the per-nucleotide score files
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresA_sorted.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresC_sorted.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresG_sorted.bedgraph')
		os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresT_sorted.bedgraph')
	os.system('rm '+options.outputFileNamePrefix+'_deepLIFTScoresMax_sorted.bedgraph')
	
def convertBedgraphsToBigwigsPerSequence(options):
	print ("Converting to per-sequence bigwigs!")
	from getDeepLIFTScoresCrossVal import bedgraphToBigwig
	bedgraphCat = None
	for i in range(options.numSequences):
		# Iterate through the sequences and make a separate bigwig for each
		outputFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresMax", str(i) + ".bedgraph"])
		if i == 0:
			# Initialize the cat bedtool
			bedgraphCat = bt.BedTool(outputFileName)
		else:
			# Add the current bedgraph to the existing concatenated bedgraph
			bedgraphCat = bedgraphCat.cat(outputFileName, postmerge=False).sort()
		# Convert bedgraphs to bigwigs, using a separate file for each sequence
		bedgraphToBigwig(outputFileName, options.genomeName, options.outputFileNamePrefix + "_" + str(i) + ".bw")
	bedgraphMeanOutputFileName = options.outputFileNamePrefix + "_mean.bedgraph"
	bedgraphMean = bedgraphCat.groupby(g=[1, 2, 3], c=4, o="mean").sort()
	bedgraphMean.saveas(bedgraphMeanOutputFileName)
	bedgraphStdOutputFileName = options.outputFileNamePrefix + "_std.bedgraph"
	bedgraphStd = bedgraphCat.groupby(g=[1, 2, 3], c=4, o="stdev").sort()
	bedgraphStd.saveas(bedgraphStdOutputFileName)
	bedgraphToBigwig(bedgraphMeanOutputFileName, options.genomeName, options.outputFileNamePrefix + "_mean.bw")
	bedgraphToBigwig(bedgraphStdOutputFileName, options.genomeName, options.outputFileNamePrefix + "_std.bw")
	for i in range(options.numSequences):
		# Remove the bedgraph files because they take up a lot of space
		outputFileName = "_".join([options.outputFileNamePrefix, "deepLIFTScoresMax", str(i) + ".bedgraph"])
		os.system('rm '+ outputFileName)
	os.system('rm '+ bedgraphMeanOutputFileName)
	os.system('rm '+ bedgraphStdOutputFileName)

def getDeepLIFTScores(options):
	# Get the DeepLIFT scores for a set of regions from a model in bedgraph format
	sys.path.insert(0,options.codePath + "/deeplearning")
	from sequenceOperations import makeSummitPlusMinus
	from getDeepLIFTGrammars import getTruePositiveIndices, getTrueNegativeIndices
	from keras.models import model_from_json
	sys.path.insert(0,options.codePath + "/deeplift")
	import deeplift.conversion.keras_conversion as kc
	from deeplift.util import get_hypothetical_contribs_func_onehot
	import deeplift.blobs.activations as activations
	model = model_from_json(open(options.modelArchitectureFileName).read())
	peakInfoFileNameElements = options.peakInfoFileName.split(".")
	peakInfoFileNamePrefix = ".".join(peakInfoFileNameElements[0:-2])
	taskNum = 0
	model.load_weights(options.modelWeightsFileName)
	#Covnert the model
	deeplift_model = kc.convert_sequential_model(model, nonlinear_mxts_mode=activations.NonlinearMxtsMode.Rescale)
	#Compile the functions to compute the contributions and multipliers - the multipliers are analogous to the gradients
	target_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)
	target_multipliers_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0)
	#Load the input data
	trueLabelIndices = np.empty((1,1))
	trueLabelRegions = bt.BedTool()
	X = None
	Y = None
	trueLabelIndices = None
	inputIsFasta = False
	summitPlusMinus = None
	if (not options.differentialPeaks):
		# The peaks are not differential peaks
		assert (not options.negSet)
		from sequenceOperations import makePositiveSequenceInputArraysFromNarrowPeaks, makePositiveSequenceInputArraysFromFasta
		if ((not options.peakInfoFileName.endswith("fa")) and (not options.peakInfoFileName.endswith("fasta"))) and \
			(not options.peakInfoFileName.endswith("fna")):
			# The input file is a narrowPeak file
			optimalBedFileName = peakInfoFileNamePrefix + ".bed"
                	summitPlusMinus =\
                        	makeSummitPlusMinus(optimalBedFileName, createOptimalBed=False, dataShape=(1,4,options.sequenceLength), bedFilegzip=False, \
					chroms=options.chroms, maxPeakLength=options.maxPeakLength)
			X, Y =\
				makePositiveSequenceInputArraysFromNarrowPeaks(options.peakInfoFileName, options.genomeFileName, createOptimalBed=False, \
					dataShape=(1,4,options.sequenceLength), chroms=options.chroms, multiMode=False, maxPeakLength=options.maxPeakLength)
		else:
			# The input is a fasta file
			inputIsFasta = True
			X, Y = makePositiveSequenceInputArraysFromFasta(options.peakInfoFileName, dataShape=(1,4,options.sequenceLength))
		#Load the input data
		trueLabelIndices = getTruePositiveIndices(X, Y, model, taskNum)
	else:
		# The peaks are differential peaks, so only the summits have been included
		summitPlusMinus = makeSummitPlusMinus(options.peakInfoFileName, createOptimalBed=False, dataShape=(1,4,options.sequenceLength), summitCol=1, startCol=None)
		from sequenceOperations import makeSequenceInputArraysFromDifferentialPeaks
		X, Y, _, _, _, indices =\
			makeSequenceInputArraysFromDifferentialPeaks(options.DESeq2OutputFileName, options.genomeFileName, options.peakInfoFileName, \
				(1,4,options.sequenceLength), createOptimalBed=False, backgroundSummitPresent=False, backgroundSummitOnly=True, createModelDir=False, \
				chroms=options.chroms, bigWigFileNames=[], multiMode=False, streamData=False, dataFileName="", RC=False)
		if (not options.negSet):
			# Run deepLIFT on the positives
			trueLabelIndices = getTruePositiveIndices(X, Y, model, taskNum)
		else:
			# Run deepLIFT on the negatives
			trueLabelIndices = getTrueNegativeIndices(X, Y, model, taskNum)
	data = None
	if not options.includeIncorrect:
		# Get only the deepLIFT scores for the correctly predicted examples
		data = X[trueLabelIndices]
		print("Last true label index: " + str(trueLabelIndices[-1]))
	elif options.negSet:
		# Get the deepLIFT scores for the negative examples
		data = X[Y == 0, :, :, :]
	else:
		# Get the deepLIFT scores for the positive examples
		data = X[Y == 1, :, :, :]
	# Obtain the deepLIFT scores for the data
	contribs_datas = [np.array(target_contribs_func(task_idx=i, input_data_list=[data], batch_size=500, progress_update=5000)) for i in [0]]
	if options.negSet:
		# Negate the deepLIFT scores
		contribs_datas = [0 - cd for cd in contribs_datas]
	print(np.shape(contribs_datas[0]))
    	# Save the deepLIFT scores for the data
	if (not options.hdf5Output) and (not inputIsFasta):
		# Save the deepLIFT scores as text files
		makeDeepLIFTScoreFiles(options, contribs_datas)
		trueLabelRegions = summitPlusMinus.at(indices[trueLabelIndices])
        	trueLabelRegionsFileName = options.outputFileNamePrefix + "_summitsPlusMinus500bp_trueLabels.bed"
        	trueLabelRegions.saveas(trueLabelRegionsFileName)
		# Convert the deepLIFT score files into bedgraph files
		if not options.separateFilePerSequence:
			# Make one bedgraph file for each nucleotide and its RC
			makeBedGraphFromPositionScoresAll(options, trueLabelRegionsFileName)
			# Sort the bedgraph files and keep the top score for any position with multiple scores
			sortBedgraphFiles(options)
			# Convert the sorted bedgraph files to bigwig files
			convertBedgraphsToBigwigs(options)
		else:
			# Make a separate bedgraph file for every sequence
			assert(options.maxOnly)
			makeBedGraphFromPositionScoresPerSequenceAll(options, trueLabelRegionsFileName)
			# Convert the sorted bedgraph files to bigwig files
			convertBedgraphsToBigwigsPerSequence(options)
	else:
                assert(options.getHypotheticalContribs)
	if options.getHypotheticalContribs:
		# Get the hypothetical contributions
		hypothetical_contribs_func = get_hypothetical_contribs_func_onehot(target_multipliers_func)
		hypothetical_contribs_datas =\
			[np.array(hypothetical_contribs_func(task_idx=i, input_data_list=[data], input_references_list=[np.zeros(data.shape)], \
				batch_size=500, progress_update=5000)) for i in [0]]
		assert(options.hdf5Output)
		# Save the deepLIFT scores as hdf5 files
                contribs_datasReshape = np.zeros((contribs_datas[0].shape[0], contribs_datas[0].shape[3], contribs_datas[0].shape[2]))
		hypothetical_contribs_datasReshape = np.zeros(contribs_datasReshape.shape)
		for i in range(contribs_datas[0].shape[0]):
			# Iterate through the examples and re-format the importance and hypothetical importance scores for each
			for j in range(contribs_datas[0].shape[3]):
				# Iterate through the length of the sequence and re-format the channels
				for k in range (contribs_datas[0].shape[2]):
					# Iterate through the channels and fill in the re-fromatted imporgance and hypothetical importance sores
					contribs_datasReshape[i,j,k] = contribs_datas[0][i,0,k,j]
					hypothetical_contribs_datasReshape[i,j,k] = hypothetical_contribs_datas[0][i,0,k,j]
		outputFileName = options.outputFileNamePrefix+"_deepLIFTScores.h5"
                f = h5py.File(outputFileName)
		g = f.create_group("contrib_scores")
		g.create_dataset("task"+str(0), data=contribs_datasReshape)
		g = f.create_group("hyp_contrib_scores")
		g.create_dataset("task"+str(0), data=hypothetical_contribs_datasReshape)
		f.close()
		
if __name__ == "__main__":
	options = parseArgument()
	getDeepLIFTScores(options)
