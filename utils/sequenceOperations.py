import sys
import os
import subprocess
import gzip
import numpy as np
from itertools import izip
import h5py
import pybedtools as bt
from Bio import SeqIO
from Bio.Seq import Seq
from mergedPeaksSummitAndMakeContinuousMat import convertStartToSummit, convertSummitToStart, swapColumns, show_value
import pyximport
importers = pyximport.install()
from one_hot_encode import one_hot_encode as one_hot_encode_cython
pyximport.uninstall(*importers)

def one_hot_encode(sequence):
	encoded_sequence = np.zeros((4, len(sequence)), dtype=np.int8)
	one_hot_encode_cython(sequence, encoded_sequence)
	numNs = len(sequence) - np.sum(encoded_sequence)
	return encoded_sequence, numNs
		
def reverse_complement(encoded_sequences):
	# Because the encoding is A, C, G, T in that order, can just reverse each sequence along both axes.
	# Also in sequenceOperations.py
	return encoded_sequences[..., ::-1, ::-1]

def addSequencesForComparison(sequenceArrayLMat, sequenceArrayLPat, sequenceArraySMat, sequenceArraySPat, dataShape, nucleotidesTogether=False, perBaseTrackFileNames=[]):
	# Create a sequence input for a set of 4 sequences
	# Put the maternal allele first
	sequenceArrayLMatFirst = np.vstack((sequenceArrayLMat, sequenceArrayLPat))
	sequenceArraySMatFirst = np.vstack((sequenceArraySMat, sequenceArraySPat))
	if nucleotidesTogether:
		# Put the As, Cs, Gs, and Ts together instead of putting each parent together
		sequenceArrayLMatFirst = np.vstack((sequenceArrayLMat[0,:],sequenceArrayLPat[0,:],sequenceArrayLMat[1,:],sequenceArrayLPat[1,:],sequenceArrayLMat[2,:],sequenceArrayLPat[2,:],sequenceArrayLMat[3,:],sequenceArrayLPat[3,:]))
		sequenceArraySMatFirst = np.vstack((sequenceArraySMat[0,:],sequenceArraySPat[0,:],sequenceArraySMat[1,:],sequenceArraySPat[1,:],sequenceArraySMat[2,:],sequenceArraySPat[2,:],sequenceArraySMat[3,:],sequenceArraySPat[3,:]))
	sequenceArrayStackLSMatFirst = np.vstack((sequenceArrayLMatFirst, sequenceArraySMatFirst))
	sequenceArrayReshapeLSMatFirst = np.reshape(sequenceArrayStackLSMatFirst, dataShape)
	sequenceArrayStackSLMatFirst = np.vstack((sequenceArraySMatFirst, sequenceArrayLMatFirst))
	sequenceArrayReshapeSLMatFirst = np.reshape(sequenceArrayStackSLMatFirst, dataShape)
	# Put the paternal allele first
	sequenceArrayLPatFirst = np.vstack((sequenceArrayLPat, sequenceArrayLMat))
	sequenceArraySPatFirst = np.vstack((sequenceArraySPat, sequenceArraySMat))
	if nucleotidesTogether:
		# Put the As, Cs, Gs, and Ts together instead of putting each parent together
		sequenceArrayLPatFirst = np.vstack((sequenceArrayLPat[0,:],sequenceArrayLMat[0,:],sequenceArrayLPat[1,:],sequenceArrayLMat[1,:],sequenceArrayLPat[2,:],sequenceArrayLMat[2,:],sequenceArrayLPat[3,:],sequenceArrayLMat[3,:]))
		sequenceArraySPatFirst = np.vstack((sequenceArraySPat[0,:],sequenceArraySMat[0,:],sequenceArraySPat[1,:],sequenceArraySMat[1,:],sequenceArraySPat[2,:],sequenceArraySMat[2,:],sequenceArraySPat[3,:],sequenceArraySMat[3,:]))
	sequenceArrayStackLSPatFirst = np.vstack((sequenceArrayLPatFirst, sequenceArraySPatFirst))
	sequenceArrayReshapeLSPatFirst = np.reshape(sequenceArrayStackLSPatFirst, dataShape)
	sequenceArrayStackSLPatFirst = np.vstack((sequenceArraySPatFirst, sequenceArrayLPatFirst))
	sequenceArrayReshapeSLPatFirst = np.reshape(sequenceArrayStackSLPatFirst, dataShape)
	return [sequenceArrayReshapeLSMatFirst, sequenceArrayReshapeSLMatFirst, sequenceArrayReshapeLSPatFirst, sequenceArrayReshapeSLPatFirst]

def loadPerBaseTracks(perBaseTrackFileNames):
	# Load the per base tracks
	# Also in sequenceOperations.py
	perBaseTracks = []
	if perBaseTrackFileNames:
		# Load the per base track files
		perBaseTracks = [np.loadtxt(pbtfn) for pbtfn in perBaseTrackFileNames]
	return perBaseTracks
	
def createPerBaseTracksMat(perBaseTracks, width, sampleCount, divisor):
	# Create a matrix with the per base tracks for the current sample
	# ASSUMES THAT sampleCount IS A MULTIPLE OF divisor
	# Also in sequenceOperations.py
	perBaseTracksIndex = sampleCount / divisor
	perBaseTracksMat = np.empty((0, width))
	for pbt in perBaseTracks:
		# Iterate through the per base 
		perBaseTracksMat = np.vstack((perBaseTracksMat, pbt[perBaseTracksIndex, :]))
	return perBaseTracksMat
	
def makeMultiModedData(allData, dataShape, numPerBaseTracks):
	# Convert data into the format for multi-moding
	# ASSUMES per-base tracks are 1 high
	# Also in sequenceOperations.py
	assert(numPerBaseTracks > 0)
	allDataList = []
	allDataList.append(allData[:,:,0:dataShape[1],:])
	for i in range(numPerBaseTracks):
		# Iterate through the per-base tracks and add the data for that track to the list
		allDataList.append(allData[:,:,dataShape[1] + i - 1:dataShape[1] + i,:])
	return allDataList

def convertFastaFileToSequencesFile(fastaFileName):
	# Convert a fasta file to a sequences file
	# Also in sequenceOperations.py
	numSequences = 0
	sequencesFileName = ".".join(fastaFileName.split(".")[-0:-1]) + "_sequences.txt"
	sequencesFile = open(sequencesFileName, 'w+')
	sequenceIDs = []
	for record in SeqIO.parse(fastaFileName, "fasta"):
		sequencesFile.write(str(record.seq) + "\n")
		numSequences = numSequences + 1
		sequenceIDs.append(record.id)
	sequencesFile.close()
	return sequencesFileName, numSequences, sequenceIDs
		
def makeSequenceInputArraysNoLabels(sequenceFileName, dataShape, numSequences, perBaseTrackFileNames=[], multiMode=False, maxFracNs = 1.0):
	# Convert each sequence into a numpy array, but do not load any labels/signals files
	# ASSUMES THAT THE SEQUENCES ARE LISTS AND NOT IN FASTA FORMAT
	# Also in sequenceOperations.py
	sequenceFile = open(sequenceFileName)
	perBaseTracks = loadPerBaseTracks(perBaseTrackFileNames)
	channel1 = dataShape[0];
	channel2 = dataShape[1] + len(perBaseTracks);
	channel3 = dataShape[2];
	allData = np.zeros((numSequences*2, channel1, channel2, channel3), dtype=np.int8);
	if perBaseTracks:
		# There are additional per-base tracks that might not be ints
		allData = np.zeros((numSequences*2, channel1, channel2, channel3), dtype=np.float16);
	sampleCount = 0
	skippedIndices = []
	totalNs = 0
	for sequence in sequenceFile:
		# Iterate through the fasta sequences and create the alphabets for the sequence and the reverse complement of each
		perBaseTracksMat = createPerBaseTracksMat(perBaseTracks, channel3, sampleCount, 2)
		sequenceArray, numNs = one_hot_encode(sequence.strip())
		sequenceFracNs = float(numNs)/float(dataShape[2])
		if sequenceFracNs > maxFracNs:
			# The percentage of N's in the current sequence is too high
			print("This sequence has too high of a percentage of N's: " + sequence + " " + str(sequenceFracNs))
			numSequences = numSequences - 1
			skippedIndices.append(sampleCount/2)
			continue
		if sequenceArray.shape[1] != dataShape[2]:
			# The current sequences is the wrong length, so skip it
			print("This sequence is the wrong length: " + sequence)
			skippedIndices.append(sampleCount/2)
			numSequences = numSequences - 1
			continue
		totalNs = totalNs + numNs
		sequenceArrayReshape = np.reshape(np.vstack((sequenceArray, perBaseTracksMat)), (channel1, channel2, channel3))
		allData[sampleCount,:,:,:] = sequenceArrayReshape
		sampleCount = sampleCount + 1
		# Repeat for the reverse complement
		sequenceArrayRC = reverse_complement(sequenceArray)
		sequenceArrayReshapeRC = np.reshape(np.vstack((sequenceArrayRC, perBaseTracksMat)), (channel1, channel2, channel3))
		allData[sampleCount,:,:,:] = sequenceArrayReshapeRC
		sampleCount = sampleCount + 1
	assert (sampleCount == numSequences*2)
	fracNs = float(totalNs)/float(dataShape[2] * numSequences)
	print("The fraction of Ns is: " + str(fracNs))
	sequenceFile.close()
	if multiMode:
		# Re-format the data for multi-moding
		return makeMultiModedData(allData, dataShape, len(perBaseTrackFileNames))
	return allData, skippedIndices
	
def createBedToolForFilteredList(regionList, createBedFilt, chroms, bedFiltFileName=None):
	# Get the BedTool for a filtered set of peaks
	# Also in sequenceOperations.py
	if createBedFilt:
		# Create a bed file for a filtered set of peaks
		regionListFilt = bt.BedTool([region for region in regionList if region[0] in chroms])
		regionListFilt.saveas(bedFiltFileName)
	if bedFiltFileName:
		# Save the region list to a file
		regionListFilt = bt.BedTool(bedFiltFileName)
	bt.helpers.cleanup()
	return regionListFilt
	
def defineInterval(r, halfWindowSize, summitPresent, windowSizeOdd=False):
        # Create an interval that the CNN can take from a region from a bed file
	# Also in sequenceOperations.py
        chrom = show_value(r[0])
        if summitPresent:
                # Convert the region to a summit-centered interval
                start = int(show_value(r[1])) + int(show_value(r[9])) - halfWindowSize
                if windowSizeOdd:
                        # Subtract 1 from the start
                        start = start - 1
                end = int(show_value(r[1])) + int(show_value(r[9])) + halfWindowSize
                return [chrom, start, end]
        else:
                # Use the centers of the peaks instead of summits
                start = int(show_value(r[1])) + int(round((float(show_value(r[2])) - float(show_value(r[1])))/2.0)) - halfWindowSize
                if windowSizeOdd:
                        # Subtract 1 from the start
                        start = start - 1
                end = int(show_value(r[1])) + int(round((float(show_value(r[2])) - float(show_value(r[1])))/2.0)) + halfWindowSize
                return [chrom, start, end]

def createSetForDeepLearning(genomeFileName, regionList, peakFileNamePrefix, halfWindowSize, summitPresent=True, maxPeakLength=None, \
	chromSizesFileName=None, windowSizeOdd=False, chromEdgeDistLimit=0):
	# Also in sequenceOperations.py
        chromSizesDict = None
        if chromSizesFileName != None:
                # Create a dictionary mapping chromosomes to their sizes
                chromSizesFile = open(chromSizesFileName)
                chromSizesDict = {}
                for line in chromSizesFile:
                        # Iterate through the chromosome sizes and make an entry in the dictionary for each
                        lineElements = line.strip().split("\t")
                        chromSizesDict[lineElements[0]] = int(lineElements[1])
        intervalList = []
        regionListFiltList = []
        for r in regionList:
                # Convert the list of regions into intervals
                [chrom, start, end] = defineInterval(r, halfWindowSize, summitPresent, windowSizeOdd)
                if start < chromEdgeDistLimit:
                        # Do not use the current region because it is too close to the start of the chromosome
                        print ("Start < chromEdgeDistLimit for region: " + str(r))
                        continue
                if chromSizesDict != None:
                        # Check if the current region is too close to the end of the chromosome
                        if chrom not in chromSizesDict:
                                # The current chromosome is not in the dictionary, so skip it
                                print "Chromosome " + chrom + " is not in the list of chromosomes"
                                continue
                        if end > chromSizesDict[chrom] - chromEdgeDistLimit:
                                # Do not use the current region because it is too close to the end of the chromosome
                                print ("End greater than chromosome length - chromEdgeDistLimit for region: " + str(r))
                                continue
                if (maxPeakLength != None) and (int(round(float(show_value(r[2])) - float(show_value(r[1])))) > maxPeakLength):
                        # The current region is too log, so skip it
                        continue
                regionListFiltList.append(r)
                intervalList.append(bt.Interval(chrom, start, end, show_value(r[4])))
        regionListFilt = bt.BedTool(regionListFiltList)
        summitPlusMinus = bt.BedTool(intervalList)
        fastaFileName = None
        if not windowSizeOdd:
                # Do not add 1 to the half window size in the name of the fasta file
                fastaFileName = ".".join([peakFileNamePrefix, "plusMinus" + str(halfWindowSize) + "bp", "fa"])
        else:
                # Add 1 to the half window size in the name of the fasta file
                fastaFileName = ".".join([peakFileNamePrefix, "plusMinus" + str(halfWindowSize + 1) + "bp", "fa"])
        fasta = summitPlusMinus.sequence(fi = genomeFileName, fo = fastaFileName)
        return summitPlusMinus, fastaFileName, regionListFilt
	
def addSequenceToArray(channel1, channel2, channel3, sequenceRecord, perBaseTracks, allData, sampleCount, perBaseTracksDivisor=2):
	# Add a sequence and its reverse complement to a numpy array
	# Also in sequenceOperations.py
	perBaseTracksMat = createPerBaseTracksMat(perBaseTracks, channel3, sampleCount, perBaseTracksDivisor)
	sequenceArray, numNs = one_hot_encode(str(sequenceRecord.seq).strip())
	sequenceArrayReshape = np.reshape(np.vstack((sequenceArray, perBaseTracksMat)), (channel1, channel2, channel3))
	allData[sampleCount,:,:,:] = sequenceArrayReshape
	sampleCount = sampleCount + 1
	# Repeat for the reverse complement
	sequenceArrayRC = reverse_complement(sequenceArray)
	sequenceArrayReshapeRC = np.reshape(np.vstack((sequenceArrayRC, perBaseTracksMat)), (channel1, channel2, channel3))
	allData[sampleCount,:,:,:] = sequenceArrayReshapeRC
	sampleCount = sampleCount + 1
	return [allData, sampleCount]

def addSequenceToArrayNoRC(channel1, channel2, channel3, sequenceRecord, perBaseTracks, allData, sampleCount):
	# Add a sequence and its reverse complement to a numpy array
	# Also in sequenceOperations.py
	perBaseTracksMat = createPerBaseTracksMat(perBaseTracks, channel3, sampleCount, 2)
	sequenceArray, numNs = one_hot_encode(str(sequenceRecord.seq).strip())
	sequenceArrayReshape = np.reshape(np.vstack((sequenceArray, perBaseTracksMat)), (channel1, channel2, channel3))
	allData[sampleCount,:,:,:] = sequenceArrayReshape
	sampleCount = sampleCount + 1
	return [allData, sampleCount]
	
def saveDataToHDF5(dataFileName, data, labels):
	# Save the data numpy arrays as an hdf5 file
	# Based on code by Anna Shcherbina
	f=h5py.File(dataFileName,'w')
	dataFileNameElements = dataFileName.split(".")
	dsetName = ".".join(dataFileNameElements[0:-1])
	print("dsetName is: " + dsetName)
	dset_X=f.create_dataset(dsetName+"X",data=data)
	dset_Y=f.create_dataset(dsetName+"Y",data=labels)
	f.flush()
	f.close()
	print ("Data hdf5 created successfully!")
	
def loadDataFromHDF5(dataFileName, indexes):
	# Load data from an hdf5 file
	dataFileNameElements = dataFileName.split(".")
	dsetName = ".".join(dataFileNameElements[0:-1])
	x_key = dsetName+"X" 
	y_key = dsetName+"Y" 
	inputFile = h5py.File(self.dataFileName,'r')
	X = np.asarray(inputFile[x_key])[indexes,:,:,:]
	Y = np.asarray(inputFile[y_key])[indexes]
	return X, Y
	
def convertQTLFileToBed(QTLFileName):
	# Convert a QTL file into a bed file
	QTLBedFileName = QTLFileName + ".bed"
	QTLFile = open(QTLFileName)
	QTLBedFile = open(QTLBedFileName, 'w+')
	for line in QTLFile:
		# Iterate through the lines of the QTL file and convert each line into bed format
		lineElements = line.strip().split("\t")
		QTLBedFile.write("\t".join([lineElements[0], str(int(lineElements[1]) - 1), lineElements[1]]) + "\n")
	QTLFile.close()
	QTLBedFile.close()
	return QTLBedFileName
	
def getQTLInfo(QTLLine):
	# Get the chromosome, position, positive direction allele, and negative direction allele from a line in a QTL file
	if QTLLine == "":
		# At the end of the QTL file
		return ["", 0, "", ""]
	QTLLineElements = QTLLine.strip().split("\t")
	position = int(QTLLineElements[1])
	return [QTLLineElements[0], position, QTLLineElements[2], QTLLineElements[3]]
	
def getRecordInfo(recordid):
	# Get the chromosome and position information from a biopython Seq record
	recordidElements = recordid.split(":")
	recordidPositionElements = recordidElements[1].split("-")
	return [recordidElements[0], int(recordidPositionElements[0]), int(recordidPositionElements[1])]
	
def createExamplesFromQTLFile(QTLFileName, fastaFileName, channel1, channel2, channel3, perBaseTracks, allData, sampleCount):
	# Create numpy arrays based on a QTL file
	print("Creating examples from: " + QTLFileName)
	QTLFile = open(QTLFileName)
	QTLInfo = getQTLInfo(QTLFile.readline())
	for record in SeqIO.parse(fastaFileName, "fasta"):
		# Iterate through the positive fastas and make a numpy array for each
		recordInfo = getRecordInfo(record.id.strip())
		recordSeq = str(record.seq)
		while QTLInfo[0] < recordInfo[0]:
			# Iterate through the QTLs until one that is on the correct chromosome is reached
			QTLInfo = getQTLInfo(QTLFile.readline())
			if QTLInfo[0] == "":
				# At the end of the QTL file, so stop
				break
		while (QTLInfo[0] == recordInfo[0]) and (QTLInfo[1] < recordInfo[1]):
			# Iterate through the QTLs until one that is at the correct position is reached
			QTLInfo = getQTLInfo(QTLFile.readline())
			if QTLInfo[0] == "":
				# At the end of the QTL file, so stop
				break
		intersectionFound = False
		openRecordSeq = recordSeq
		closedRecordSeq = recordSeq
		while (QTLInfo[0] == recordInfo[0]) and ((QTLInfo[1] >= recordInfo[1]) and (QTLInfo[1] < recordInfo[2])):
			# Iterate through the QTLs until one that is on the correct chromosome is reached
			intersectionFound = True
			QTLPosition = QTLInfo[1] - recordInfo[1]
			openRecordSeq = recordSeq[0:QTLPosition] + QTLInfo[2] + recordSeq[QTLPosition+1:len(recordSeq)]
			closedRecordSeq = recordSeq[0:QTLPosition] + QTLInfo[3] + recordSeq[QTLPosition+1:len(recordSeq)]
			QTLInfo = getQTLInfo(QTLFile.readline())
			if QTLInfo[0] == "":
				# At the end of the QTL file, so stop
				break
		if intersectionFound:
			# Add the sequence to the dataset
			openRecord = record
			openRecord.seq = Seq(openRecordSeq)
			[allData, sampleCount] =\
				addSequenceToArray(channel1, channel2, channel3, openRecord, perBaseTracks, allData, sampleCount, perBaseTracksDivisor=4)
			closedRecord = record
			closedRecord.seq = Seq(closedRecordSeq)
			[allData, sampleCount] =\
				addSequenceToArray(channel1, channel2, channel3, closedRecord, perBaseTracks, allData, sampleCount, perBaseTracksDivisor=4)
	QTLFile.close()
	return [allData, sampleCount]
	
def makeSequenceInputArraysFromNarrowPeaksSNPs(QTLFileName, nonQTLFileName, genomeFileName, backgroundFileName, dataShape, \
			createOptimalBed=False, backgroundSummitPresent=False, backgroundSummitOnly=False, createModelDir=False, \
			chroms=["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr2", "chr20", "chr22", \
				"chr3", "chr4", "chr5", "chr6", "chr7", "chr9", "chrX"], \
			bigWigFileNames=[], multiMode=False, streamData=False, dataFileName=""):
	# Convert each peak into a numpy array, where pos. direction alleles and no effect SNPs are positives and neg. direction alleles are negatives
	# ASSUMES THAT QTL FILES ARE IN THE FORM CHROM., POSITION, POSITIVE EFFECT ALLELE, NEGATIVE EFFECT ALLELE
	# ASSUMES THAT QTL FILES ARE SORTED BY CHROM., POSITION
	# ASSUMES THAT BACKGROUND PEAKS ARE MERGED
	# ASSUMES THAT THIS IS FOR BINARY CLASSIFICATION
	# ASSUMES THAT SNPs IN THE SAME PEAK ARE IN LD SUCH THAT THE OPEN/REFERENCE ALLELES ARE ALWAYS TOGETHER
	backgroundRegionList = None
	if not backgroundSummitOnly:
		# The background dataset has peaks, not just summits
		backgroundRegionList = bt.BedTool(backgroundFileName).filter(lambda x: x.chrom in chroms).sort()
	else:
		# Create regions from the summits
		backgroundRegionList =\
			makeSummitPlusMinus(backgroundFileName, createOptimalBed=False, dataShape=dataShape, summitCol=1, startCol=None).filter(lambda x: \
				x.chrom in chroms).sort()
	backgroundFileNameElements = backgroundFileName.split(".")
	backgroundFileNamePrefix = ".".join(backgroundFileNameElements[0:-1])
	modelDir = backgroundFileNamePrefix + ".KerasModels"
	if createModelDir:
		# Create a new directory where the model parameters will be written
		os.mkdir(modelDir)
	halfWindowSize = dataShape[2]/2
	summitPlusMinus, fastaFileName, _ =\
		createSetForDeepLearning(genomeFileName, backgroundRegionList, backgroundFileNamePrefix, halfWindowSize, 
			summitPresent=backgroundSummitPresent)
	perBaseTracks = []
	if bigWigFileNames:
		# There are per-base tracks, so load them
		from bigWigFeaturizeOverBedNumpy import bigWigFeaturize
		perBaseTracks = bigWigFeaturize.new(bigWigFileNames, dataShape[2], intervals=[region for region in backgroundRegionList], cache="cache")
	QTLBedFileName = convertQTLFileToBed(QTLFileName)
	nonQTLBedFileName = convertQTLFileToBed(nonQTLFileName)
	numQTLsInPeaks = backgroundRegionList.intersect(QTLBedFileName, wa=True, u=True).count()
	numNonQTLsInPeaks = backgroundRegionList.intersect(nonQTLBedFileName, wa=True, u=True).count()
	os.remove(QTLBedFileName)
	os.remove(nonQTLBedFileName)
	labels = np.concatenate((np.tile(np.array([1,1,0,0]), numQTLsInPeaks), np.ones(4 * numNonQTLsInPeaks)), axis = 0)
	channel1 = dataShape[0];
	channel2 = dataShape[1] + len(bigWigFileNames);
	channel3 = dataShape[2];
	allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.int8)
	if bigWigFileNames:
		# There are additional per-base tracks that might not be ints
		allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.float16)
	sampleCount = 0
	[allData, sampleCount] =\
		createExamplesFromQTLFile(QTLFileName, fastaFileName, channel1, channel2, channel3, perBaseTracks, allData, sampleCount)
	[allData, sampleCount] =\
		createExamplesFromQTLFile(nonQTLFileName, fastaFileName, channel1, channel2, channel3, perBaseTracks, allData, sampleCount)
	os.remove(fastaFileName)
	assert (sampleCount == labels.shape[0])
	print ("The number of positives is: " + str(np.count_nonzero(labels)))
	print ("The number of negatives is: " + str(labels.shape[0] - np.count_nonzero(labels)))
	if multiMode:
		# Re-format the data for multi-moding
		allDataMultiModed = makeMultiModedData(allData, dataShape, len(perBaseTrackFileNames))
		if streamData:
			# Prepare the data for streaming instead of storing it
			saveDataToHDF5(dataFileName, allDataMultiModed, labels)
		bt.helpers.cleanup()
		return allDataMultiModed, labels, modelDir
	else:
		# Do not re-format the data for multi-moding
		if streamData:
			# Prepare the data for streaming instead of storing it
			saveDataToHDF5(dataFileName, allData, labels)
		bt.helpers.cleanup()
		return allData, labels, modelDir
	
def createPositiveSetFromNarrowPeaks(optimalPeakFileName, genomeFileName, dataShape, createOptimalBed=False, createOptimalBedFilt=True, \
	maxPeakLength=None, chroms=None, chromSizesFileName=None, chromEdgeDistLimit=0):
        # Create the positive set for the deep learning model
	# Also in sequenceOperations.py
        optimalPeakFileNameElements = optimalPeakFileName.split(".")
        optimalPeakFileNamePrefix = ".".join(optimalPeakFileNameElements[0:-2])
        optimalBedFileName = optimalPeakFileNamePrefix + "_optimal.bed"
        if createOptimalBed:
                # Create a bed file for the optimal peaks
                os.system(" ".join(["zcat", optimalPeakFileName, "| grep -v chrUn | grep -v random | grep chr | sort -k1,1 -k2,2n -k3,3n -k10,10n >", \
                        optimalBedFileName]))
        else:
                os.system(" ".join(["zcat", optimalPeakFileName, "| sort -k1,1 -k2,2n -k3,3n -k10,10n >", optimalBedFileName]))
        optimalRegionList = bt.BedTool(optimalBedFileName)
        if chroms != None:
                # Filter for specific chromosomes
                optimalBedFiltFileName = optimalPeakFileNamePrefix + ".train.bed"
                optimalRegionListFilt = createBedToolForFilteredList(optimalRegionList, createOptimalBedFilt, chroms, optimalBedFiltFileName)
        else:
                # Include all of the chromosomes
                optimalRegionListFilt = optimalRegionList
        halfWindowSize = dataShape[2]/2
        windowSizeOdd = False
        if dataShape[2] % 2 > 0:
                # The window size is odd, so put an extra base on the upstream end
                windowSizeOdd = True
        summitPlusMinus, positiveFastaFileName, optimalRegionListFiltPlus =\
        	createSetForDeepLearning(genomeFileName, optimalRegionListFilt, optimalPeakFileNamePrefix, halfWindowSize, \
                	maxPeakLength=maxPeakLength, chromSizesFileName=chromSizesFileName, windowSizeOdd=windowSizeOdd, \
			chromEdgeDistLimit=0)
        return optimalPeakFileNamePrefix, optimalRegionList, optimalRegionListFiltPlus, halfWindowSize, summitPlusMinus, positiveFastaFileName
	
def makeSequenceInputArraysFromNarrowPeaks(optimalRegionListFilt, backgroundNoPeakRegionListFilt, positiveFastaFileName, negativeFastaFileName, \
	dataShape, bigWigFileNames=[], multiMode=False, streamData=False, dataFileName="", RC=True, removeFastas=True):
	# Convert each peak into a numpy array
	# ASSUMES THAT THIS IS FOR BINARY CLASSIFICATION
	positivePerBaseTracks = []
	negativePerBaseTracks = []
	if bigWigFileNames:
		# There are per-base tracks, so load them
		from bigWigFeaturizeOverBedNumpy import bigWigFeaturize
		positivePerBaseTracks = bigWigFeaturize.new(bigWigFileNames, dataShape[2], intervals=[region for region in optimalRegionListFilt], cache="cache")
		negativePerBaseTracks = bigWigFeaturize.new(bigWigFileNames, dataShape[2], intervals=[region for region in backgroundNoPeakRegionListFilt], cache="cache")
	labels = np.concatenate((np.ones(2 * optimalRegionListFilt.count()), np.zeros(2 * backgroundNoPeakRegionListFilt.count())), axis = 0)
	if (not RC):
		# Do not include the reverse complements
		labels = np.concatenate((np.ones(optimalRegionListFilt.count()), np.zeros(backgroundNoPeakRegionListFilt.count())), axis = 0)
	print ("The total number of examples is: " + str(labels.shape[0]))
	channel1 = dataShape[0];
	channel2 = dataShape[1] + len(bigWigFileNames);
	channel3 = dataShape[2];
	allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.int8)
	if bigWigFileNames:
		# There are additional per-base tracks that might not be ints
		allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.float16)
	sampleCount = 0
	for positiveRecord in SeqIO.parse(positiveFastaFileName, "fasta"):
		# Iterate through the positive fastas and make a numpy array for each
		if RC:
			# Include the reverse complements
			[allData, sampleCount] = addSequenceToArray(channel1, channel2, channel3, positiveRecord, positivePerBaseTracks, allData, sampleCount)
		else:
			# Do not include the reverse complements
			[allData, sampleCount] =\
				addSequenceToArrayNoRC(channel1, channel2, channel3, positiveRecord, positivePerBaseTracks, allData, sampleCount)
	for negativeRecord in SeqIO.parse(negativeFastaFileName, "fasta"):
		# Iterate through the positive fastas and make a numpy array for each
		if RC:
			# Include the reverse complements
			[allData, sampleCount] = addSequenceToArray(channel1, channel2, channel3, negativeRecord, negativePerBaseTracks, allData, sampleCount)
		else:
			# Do not include the reverse complements
			[allData, sampleCount] =\
				addSequenceToArrayNoRC(channel1, channel2, channel3, negativeRecord, negativePerBaseTracks, allData, sampleCount)
	if removeFastas:
		# Remove the fasta files
		os.remove(positiveFastaFileName)
		os.remove(negativeFastaFileName)
	assert (sampleCount == labels.shape[0])
	print ("The number of positives is: " + str(np.count_nonzero(labels)))
	print ("The number of negatives is: " + str(labels.shape[0] - np.count_nonzero(labels)))
	if multiMode:
		# Re-format the data for multi-moding
		allDataMultiModed = makeMultiModedData(allData, dataShape, len(perBaseTrackFileNames))
		if streamData:
			# Prepare the data for streaming instead of storing it
			saveDataToHDF5(dataFileName, allDataMultiModed, labels)
		return allDataMultiModed, labels, positiveFastaFileName, negativeFastaFileName
	else:
		# Do not re-format the data for multi-moding
		if streamData:
			# Prepare the data for streaming instead of storing it
			saveDataToHDF5(dataFileName, allData, labels)
		return allData, labels, positiveFastaFileName, negativeFastaFileName
	
def makeSequenceInputArraysFromNarrowPeaksBackgroundNegatives(optimalPeakFileName, relaxedPeakFileName, genomeFileName, backgroundFileName, dataShape, \
			createOptimalBed=False, createOptimalBedFilt=True, createNegativeBed=True, negativeBedString="negativeDNase", \
			backgroundSummitPresent=True, backgroundSummitOnly=False, createModelDir=False, \
			chroms=["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr2", "chr20", "chr22", \
				"chr3", "chr4", "chr5", "chr6", "chr7", "chr9", "chrX"], \
			bigWigFileNames=[], multiMode=False, streamData=False, dataFileName=""):
	# Convert each peak into a numpy array, where optimal peaks are positives and background peaks that do not overlap relaxed peaks are negatives
	# ASSUMES THAT THIS IS FOR BINARY CLASSIFICATION
	backgroundRegionList = None
	if not backgroundSummitOnly:
		# The background dataset has peaks, not just summits
		backgroundRegionList = bt.BedTool(backgroundFileName)
	else:
		# Create regions from the summits
		backgroundRegionList = makeSummitPlusMinus(backgroundFileName, createOptimalBed=False, dataShape=dataShape, summitCol=1, startCol=None)
	backgroundFileNameElements = backgroundFileName.split(".")
	backgroundNoTFFileNamePrefix = ".".join(backgroundFileNameElements[0:-2]) + ".NoTF"
	optimalPeakFileNamePrefix, optimalRegionList, optimalRegionListFilt, halfWindowSize, summitPlusMinus, positiveFastaFileName =\
		createPositiveSetFromNarrowPeaks(optimalPeakFileName, genomeFileName, dataShape, \
			createOptimalBed=createOptimalBed, createOptimalBedFilt=createOptimalBedFilt, chroms=chroms)
	modelDir = optimalPeakFileNamePrefix + ".KerasModels"
	if createModelDir:
		# Create a new directory where the model parameters will be written
		os.mkdir(modelDir)
	backgroundNoPeakRegionList = backgroundRegionList.subtract(bt.BedTool(relaxedPeakFileName).filter(lambda x: x[0][0:3] == "chr"), A=True)
	backgroundNoPeakRegionListFiltFileName = ".".join([optimalPeakFileNamePrefix, negativeBedString, "train.bed"])
	backgroundNoPeakRegionListFilt =\
		createBedToolForFilteredList(backgroundNoPeakRegionList, createNegativeBed, chroms, backgroundNoPeakRegionListFiltFileName)
	summitPlusMinus, negativeFastaFileName, _ =\
		createSetForDeepLearning(genomeFileName, backgroundNoPeakRegionListFilt, backgroundNoTFFileNamePrefix, halfWindowSize, 
			summitPresent=backgroundSummitPresent)
	allData, labels, positiveFastaFileName, negativeFastaFileName =\
		makeSequenceInputArraysFromNarrowPeaks(optimalRegionListFilt, backgroundNoPeakRegionListFilt, positiveFastaFileName, negativeFastaFileName, \
			dataShape, bigWigFileNames=bigWigFileNames, multiMode=multiMode, streamData=streamData, dataFileName=dataFileName)
	bt.helpers.cleanup()
	return allData, labels, positiveFastaFileName, negativeFastaFileName, modelDir

def makePositiveSequenceInputArraysFromFasta(positiveFastaFileName, dataShape=(1,4,1000), labels=np.array([]), RC=False):
        # Convert each peak into a numpy array, where each peak is inputted in fasta format
        # Also in sequenceOperations.py
        if labels.size == 0:
                # Need to get the labels
                cmd = "grep -c '>' $1 " + positiveFastaFileName
                if RC:
                        # Count the reverse complements in the labels
                        numPos = int(subprocess.check_output(cmd, shell=True).strip()) * 2
                else:
                        # Use only 1 label for each example
                        numPos = int(subprocess.check_output(cmd, shell=True).strip())
                labels = np.ones(numPos, dtype=np.int8)
        channel1 = dataShape[0];
        channel2 = dataShape[1];
        channel3 = dataShape[2];
        allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.int8)
        sampleCount = 0
        for positiveRecord in SeqIO.parse(positiveFastaFileName, "fasta"):
                # Iterate through the positive fastas and make a numpy array for each
                if RC:
                        # Include the reverse complement
                        [allData, sampleCount] = addSequenceToArray(channel1, channel2, channel3, positiveRecord, [], allData, sampleCount)
                else:
                        # Do not include the reverse complement
                        [allData, sampleCount] = addSequenceToArrayNoRC(channel1, channel2, channel3, positiveRecord, [], allData, sampleCount)
        return allData, labels

def makePositiveSequenceInputArraysFromNarrowPeaks(optimalPeakFileName, genomeFileName, \
			createOptimalBed=False, createOptimalBedFilt=True, dataShape=(1,4,1000), \
			chroms=["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr22", \
				"chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chrX"], \
			multiMode=False, RC=False, maxPeakLength=None):
	# Convert each peak into a numpy array
	optimalPeakFileNamePrefix, optimalRegionList, optimalRegionListFilt, halfWindowSize, summitPlusMinus, positiveFastaFileName =\
		createPositiveSetFromNarrowPeaks(optimalPeakFileName, genomeFileName, dataShape, \
			createOptimalBed=createOptimalBed, createOptimalBedFilt=createOptimalBedFilt, maxPeakLength=maxPeakLength, chroms=chroms)
	labels = np.ones(2 * optimalRegionListFilt.count())
	if not RC:
		# Do not include the reverse complement, so do not repeat labels
		labels = np.ones(optimalRegionListFilt.count())
	print ("The total number of examples is: " + str(labels.shape[0]))
	allData, labels = makePositiveSequenceInputArraysFromFasta(positiveFastaFileName, dataShape=dataShape, labels=labels, RC=RC)
	os.remove(positiveFastaFileName)
	if multiMode:
		# Re-format the data for multi-moding
		return makeMultiModedData(allData, dataShape, len(perBaseTrackFileNames)), labels, positiveFastaFileName, negativeFastaFileName
	return allData, labels
	
def makeSummitPlusMinus(optimalPeakFileName, createOptimalBed=False, dataShape=(1,4,1000), summitCol=9, startCol=1, 
		bedFilegzip=False, chroms=None, maxPeakLength=None):
	# Create a bed file that is the summit +/- half the window bp for each peak summit
	optimalPeakFileNameElements = optimalPeakFileName.split(".")
	optimalPeakFileNamePrefix = ".".join(optimalPeakFileNameElements[0:-2])
	optimalBedFileName = optimalPeakFileNamePrefix + ".bed"
	if createOptimalBed:
		# Create a bed file for the optimal peaks
		os.system(" ".join(["zcat", optimalPeakFileName, \
			"| grep -v chrUn | grep -v random | grep chr | sort -k1,1 -k2,2n -k3,3n -k" + str(summitCol + 1) +"," + str(summitCol+ 1) + "n >", \
			optimalBedFileName]))
	optimalBedFile = open(optimalPeakFileName)
	if bedFilegzip:
		# Open the optimal peak file using gzip
		optimalBedFile = gzip.open(optimalPeakFileName)
	optimalRegionList = [line.strip().split("\t") for line in optimalBedFile]
	halfWindowSize = dataShape[2]/2
	optimalRegionListFilt = optimalRegionList
	if maxPeakLength != None:
		# The maximum peak length has been defined, so remove regions that are too short
		optimalRegionListFiltList = []
		for r in optimalRegionList:
			# Iterate through the regions and keep those that are sufficiently short
			if int(round(float(show_value(r[2])) - float(show_value(r[1])))) <= maxPeakLength:
				# The current region is sufficiently short, so keep it
				optimalRegionListFiltList.append(r)
		optimalRegionListFilt = bt.BedTool(optimalRegionListFiltList)
	if startCol != None:
		# The start positions are in the file (narrowPeak file and not summit list)
		summitPlusMinus =\
			bt.BedTool([bt.Interval(r[0], int(r[startCol]) + int(r[summitCol]) - halfWindowSize, int(r[startCol])\
				+ int(r[summitCol]) + halfWindowSize)\
				for r in optimalRegionListFilt])
	else:
		# The start postions are not in the file (summit list and not narrowPeak file)
		summitPlusMinus =\
			bt.BedTool([bt.Interval(r[0], int(r[summitCol]) - halfWindowSize, \
				int(r[summitCol]) + halfWindowSize)\
				for r in optimalRegionListFilt])
	if chroms:
		# Filter the summitsPlusMinus list based on the chromosomes
		summitPlusMinus = createBedToolForFilteredList(summitPlusMinus, True, chroms)
	bt.helpers.cleanup()
	return summitPlusMinus
	
def makeFlankingRegions(optimalRegionList, chromSizesFileName, flankDistance=1000, separationDistance=0, filterRegionsToLength=True):
	# Make flanking regions
	# Also in sequenceOperations.py
	slopRegionList = optimalRegionList.slop(g=chromSizesFileName, b=separationDistance)
	flankingRegionListPre = slopRegionList.flank(g=chromSizesFileName, b=flankDistance)
	flankingRegionList = flankingRegionListPre.subtract(slopRegionList)
	if filterRegionsToLength:
		# Remove the flanks that have been partially removed
		flankingRegionList = flankingRegionList.filter(lambda x: len(x) == flankDistance)
	flankingRegionList = flankingRegionList.sort()
	return flankingRegionList
	
def makeSequenceInputArraysFromNarrowPeaksFlankNegatives(optimalPeakFileName, relaxedPeakFileName, genomeFileName, chromSizesFileName, dataShape, \
			createOptimalBed=False, createOptimalBedFilt=True, createNegativeBed=True, createModelDir=False, \
			chroms=["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr2", "chr20", "chr22", \
				"chr3", "chr4", "chr5", "chr6", "chr7", "chr9", "chrX"], \
			bigWigFileNames=[], multiMode=False, streamData=False, dataFileName=""):
	# Convert each peak into a numpy array, where optimal peaks are positives and flanking regions that do not overlap relaxed peaks are negatives
	# ASSUMES THAT THIS IS FOR BINARY CLASSIFICATION
	optimalPeakFileNamePrefix, optimalRegionList, optimalRegionListFilt, halfWindowSize, summitPlusMinus, positiveFastaFileName =\
		createPositiveSetFromNarrowPeaks(optimalPeakFileName, genomeFileName, dataShape, \
			createOptimalBed=createOptimalBed, createOptimalBedFilt=createOptimalBedFilt, chroms=chroms)
	modelDir = optimalPeakFileNamePrefix + ".KerasModels"
	if createModelDir:
		# Create a new directory where the model parameters will be written
		os.mkdir(modelDir)
	backgroundRegionList = makeFlankingRegions(optimalRegionList, chromSizesFileName, flankDistance=dataShape[2])
	backgroundNoPeakRegionList = backgroundRegionList.subtract(relaxedPeakFileName, A=True)
	backgroundNoPeakRegionListFiltFileName = optimalPeakFileNamePrefix + ".negativeFlank.train.bed"
	backgroundNoPeakRegionListFilt =\
		createBedToolForFilteredList(backgroundNoPeakRegionList, createNegativeBed, chroms, backgroundNoPeakRegionListFiltFileName)
	negativeFastaFileName = ".".join([optimalPeakFileNamePrefix, "flanks", "fa"])
	negativeFasta = backgroundNoPeakRegionListFilt.sequence(fi = genomeFileName, fo = negativeFastaFileName)
	allData, labels, positiveFastaFileName, negativeFastaFileName =\
		makeSequenceInputArraysFromNarrowPeaks(optimalRegionListFilt, backgroundNoPeakRegionListFilt, positiveFastaFileName, negativeFastaFileName, \
			dataShape, bigWigFileNames=bigWigFileNames, multiMode=multiMode, streamData=streamData, dataFileName=dataFileName)
	return allData, labels, positiveFastaFileName, negativeFastaFileName, modelDir
	
def makeRandomRegions(summitPlusMinus, relaxedPeakFileName, chromSizesFileName, numRandom=10):
	# Make random regions that do not include relaxed peaks
	randomRegionList = summitPlusMinus.shuffle(g=chromSizesFileName, excl=relaxedPeakFileName, chrom=True, noOverlapping=True)
	for i in range(1, numRandom):
		# Iterate through the number of random region lists to create, and create more random regions on each iteration
		randomRegionList =\
			randomRegionList.cat(summitPlusMinus.shuffle(g=chromSizesFileName, excl=relaxedPeakFileName, chrom=True, noOverlapping=True), postmerge=False)
	return randomRegionList.sort()
	
def makeSequenceInputArraysFromNarrowPeaksRandomNegatives(optimalPeakFileName, relaxedPeakFileName, genomeFileName, chromSizesFileName, dataShape, \
			createOptimalBed=False, createOptimalBedFilt=True, createNegativeBed=True, createModelDir=False, \
			chroms=["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr2", "chr20", "chr22", \
				"chr3", "chr4", "chr5", "chr6", "chr7", "chr9", "chrX"], \
			bigWigFileNames=[], multiMode=False, streamData=False, dataFileName="", numRandom=10):
	# Convert each peak into a numpy array, where optimal peaks are positives and flanking regions that do not overlap relaxed peaks are negatives
	# ASSUMES THAT THIS IS FOR BINARY CLASSIFICATION
	optimalPeakFileNamePrefix, optimalRegionList, optimalRegionListFilt, halfWindowSize, summitPlusMinus, positiveFastaFileName =\
		createPositiveSetFromNarrowPeaks(optimalPeakFileName, genomeFileName, dataShape, \
			createOptimalBed=createOptimalBed, createOptimalBedFilt=createOptimalBedFilt, chroms=chroms)
	modelDir = optimalPeakFileNamePrefix + ".KerasModels"
	if createModelDir:
		# Create a new directory where the model parameters will be written
		os.mkdir(modelDir)
	backgroundNoPeakRegionList = makeRandomRegions(summitPlusMinus, relaxedPeakFileName, chromSizesFileName, numRandom=numRandom)
	backgroundNoPeakRegionListFiltFileName = optimalPeakFileNamePrefix + ".negativeRandom.train.bed"
	backgroundNoPeakRegionListFilt =\
		createBedToolForFilteredList(backgroundNoPeakRegionList, createNegativeBed, chroms, backgroundNoPeakRegionListFiltFileName)
	negativeFastaFileName = ".".join([optimalPeakFileNamePrefix, "random", "fa"])
	negativeFasta = backgroundNoPeakRegionListFilt.sequence(fi = genomeFileName, fo = negativeFastaFileName)
	allData, labels, positiveFastaFileName, negativeFastaFileName =\
		makeSequenceInputArraysFromNarrowPeaks(optimalRegionListFilt, backgroundNoPeakRegionListFilt, positiveFastaFileName, negativeFastaFileName, \
			dataShape, bigWigFileNames=bigWigFileNames, multiMode=multiMode, streamData=streamData, dataFileName=dataFileName)
	return allData, labels, positiveFastaFileName, negativeFastaFileName, modelDir
	
def makeSequenceInputArraysFromDifferentialPeaks(DESeq2OutputFileName, genomeFileName, backgroundFileName, dataShape, \
			createOptimalBed=False, backgroundSummitPresent=False, backgroundSummitOnly=True, createModelDir=False, \
			chroms=["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr2", "chr20", "chr22", \
				"chr3", "chr4", "chr5", "chr6", "chr7", "chr9", "chrX"], \
			bigWigFileNames=[], multiMode=False, streamData=False, dataFileName="", RC=True, removeFastas=True, strictNegativeSet=False, \
			fcCutoff=-1, swapLabels=True, useDESeq2OutputFileNameForFastaFileName=False, chromEdgeDistLimit=0):
	# Convert each peak into a numpy array, where optimal peaks are positives and background peaks that do not overlap relaxed peaks are negatives
	# ASSUMES THAT THIS IS FOR BINARY CLASSIFICATION
	backgroundRegionList = None
	if not backgroundSummitOnly:
		# The background dataset has peaks, not just summits
		backgroundRegionList = bt.BedTool(backgroundFileName).filter(lambda x: x[0] in chroms)
	else:
		# Create regions from the summits
		backgroundRegionList =\
			makeSummitPlusMinus(backgroundFileName, createOptimalBed=False, dataShape=dataShape, summitCol=1, startCol=None)
	DESeq2Output = np.loadtxt(DESeq2OutputFileName)
	positiveRegionList = []
	negativeRegionList = []
	i = 0
	positiveIndicesList = []
	negativeIndicesList = []
	for backgroundRegion in backgroundRegionList:
		# Iterate through the regions and add the up-regulated ones to the positive set and the down-regulated ones to the negative set
		if backgroundRegion[0] not in chroms:
			# The current region is not on the list of chromosomes to include
			i = i + 1
			continue
		if np.isnan(DESeq2Output[i,1]) or np.isnan(DESeq2Output[i,5]):
			# The log2 fold change or the padj is nan, so skip this row
			i = i + 1
			continue
		if (DESeq2Output[i,1] < fcCutoff) and (DESeq2Output[i,5] < 0.05):
			# The current row has a large fold-change in the wild-type direction and a small p-value, so add the current region to the negative set
			negativeRegionList.append(backgroundRegion)
			negativeIndicesList.append(i)
		elif (not strictNegativeSet) and (DESeq2Output[i,1] >= 0):
			# The wild-type is not up-regulated, so add the current region to the positive set
			positiveRegionList.append(backgroundRegion)
			positiveIndicesList.append(i)
		elif strictNegativeSet and ((DESeq2Output[i,1] > fcCutoff) and (DESeq2Output[i,5] < 0.05)):
			# The wild-type is down-regulated, so add the current region to the positive set
			positiveRegionList.append(backgroundRegion)
			positiveIndicesList.append(i)
		i = i + 1
	print ("Number of positives: " + str(len(positiveRegionList)))
	print ("Number of negatives: " + str(len(negativeRegionList)))
	backgroundFileNameElements = backgroundFileName.split(".")
	halfWindowSize = dataShape[2]/2
	modelDir = ".".join(backgroundFileNameElements[0:-2]) + ".KerasModels"
	if createModelDir:
		# Create a new directory where the model parameters will be written
		os.mkdir(modelDir)
	backgroundPosFileNamePrefix = ".".join(backgroundFileNameElements[0:-2]) + ".Pos"
	backgroundNegFileNamePrefix = ".".join(backgroundFileNameElements[0:-2]) + ".Neg"
	if useDESeq2OutputFileNameForFastaFileName:
		# Use the DESeq2 file name instead of the background file name to create the names of the fasta file
		DESeq2OutputFileNameElements = DESeq2OutputFileName.split(".")
		backgroundPosFileNamePrefix = ".".join(DESeq2OutputFileNameElements[0:-1]) + ".Pos"
		backgroundNegFileNamePrefix = ".".join(DESeq2OutputFileNameElements[0:-1]) + ".Neg"
	summitPlusMinus, positiveFastaFileName, _ =\
		createSetForDeepLearning(genomeFileName, bt.BedTool(positiveRegionList), backgroundPosFileNamePrefix, halfWindowSize, 
			summitPresent=backgroundSummitPresent, chromEdgeDistLimit=chromEdgeDistLimit)	
	summitPlusMinus, negativeFastaFileName, _ =\
		createSetForDeepLearning(genomeFileName, bt.BedTool(negativeRegionList), backgroundNegFileNamePrefix, halfWindowSize, 
			summitPresent=backgroundSummitPresent, chromEdgeDistLimit=chromEdgeDistLimit)
	allData, labels, positiveFastaFileName, negativeFastaFileName =\
		makeSequenceInputArraysFromNarrowPeaks(bt.BedTool(positiveRegionList), bt.BedTool(negativeRegionList), 
			positiveFastaFileName, negativeFastaFileName, dataShape, bigWigFileNames=bigWigFileNames, multiMode=multiMode, streamData=streamData, 
			dataFileName=dataFileName, RC=RC, removeFastas=removeFastas)
	positiveIndicesList.extend(negativeIndicesList)
	if swapLabels:
		# Swap the positive and negative labels
		labels = 1 - labels
	bt.helpers.cleanup()
	return allData, labels, positiveFastaFileName, negativeFastaFileName, modelDir, np.array(positiveIndicesList)
	
def getNextSummit(regionFile):
	# Get the next summit
	regionLine = regionFile.readline()
	if regionLine == "":
		# The file is done
		return [("chrZ", float('inf')), True]
	else:
		# Get the summit from the file
		regionLineElements = regionLine.strip().split("\t")
		summit = int(regionLineElements[1]) + int(regionLineElements[9])
		return [(regionLineElements[0], summit), False]
	
def getRegionInfoLists(fileNameListFileName, summitIsOffset, chroms):
	# Get a list of regions from a file with a list of file names, a filtered list of files with the proper chroms., and the filtered list lengths
	# ASSUMES THAT EACH FILTERED LIST IS NON-EMPTY
	print ("Getting region information")
	fileNameListFile = open(fileNameListFileName)
	regionFileList = []
	regionFileNameList = []
	currentSummitList = []
	atRegionFileEnds = []
	for line in fileNameListFile:
		# Iterate through the lines of the file with file names and open each file
		fileName = line.strip()
		if summitIsOffset:
			# Convert the starts to summits before sorting
			regions = bt.BedTool(fileName).filter(lambda x: x.chrom in chroms).each(convertStartToSummit).sort().\
				each(convertSummitToStart)
		else:
			# Swap summits with starts before sorting
			regions = bt.BedTool(fileName).filter(lambda x: x.chrom in chroms).each(swapColumns).sort().each(swapColumns)
		fileNameElements = fileName.split(".")
		regionFileName = ".".join(fileNameElements[0:-3]) + ".filt.narrowPeak"
		regions.saveas(regionFileName)
		regionFileNameList.append(regionFileName)
		regionFile = open(regionFileName)
		regionFileList.append(regionFile)
		currentSummitList.append(getNextSummit(regionFileList[-1])[0])
		atRegionFileEnds.append(False)
	print("Region information lists have been made")
	bt.helpers.cleanup()
	return regionFileList, regionFileNameList, currentSummitList, atRegionFileEnds
	
def setAmbiguousPeaks(relaxedPeakFileListFileName, meanSummitsFileName, binaryMatFileName, binaryMatAmigFileName, chroms, summitDistCutoff=50):
	# Set the ambiguous peaks to have -1 labels, where ambigusous peaks are non-optimal with summits within summitDistCutoff of a merged peak
	print("Labeling ambiguous peaks")
	binaryMat = np.loadtxt(binaryMatFileName, dtype=np.int8)
	regionFileList, regionFileNameList, currentSummitList, atRegionFileEnds = getRegionInfoLists(relaxedPeakFileListFileName, True, chroms)
	meanSummitsFile = open(meanSummitsFileName)
	for j in range(binaryMat.shape[0]):
		# Iterate through the merged peaks and create a -1 label for each that overlaps an ambiguous peak and is not an optimal peak
		meanSummitElements = meanSummitsFile.readline().strip().split("\t")
		meanSummitChrom = meanSummitElements[0]
		meanSummitPosition = int(meanSummitElements[1])
		for i in range(binaryMat.shape[1]):
			# Iterate through the tasks and set the merged peak value to -1 if the task is only a relaxed peak but not an optimal peak
			if binaryMat[j,i] == 1:
				# The current peak is an optimal peak
				continue
			else:
				# The current peak is not an optimal peak
				currentSummit = currentSummitList[i]
				while meanSummitChrom > currentSummit[0]:
					# Iterate through the relaxed peaks until a peak on the correct chromosome has been reached
					[currentSummitList[i], atRegionFileEnds[i]] = getNextSummit(regionFileList[i])
					currentSummit = currentSummitList[i]
					if atRegionFileEnds[i]:
						# At the end of the relaxed peak file, so stop
						break
				while (meanSummitChrom == currentSummit[0]) and \
					(meanSummitPosition > currentSummit[1] + summitDistCutoff):
					# Iterate through the relaxed peaks until a peak at the correct location has been reached
					[currentSummitList[i], atRegionFileEnds[i]] = getNextSummit(regionFileList[i])
					currentSummit = currentSummitList[i]
					if atRegionFileEnds[i]:
						# At the end of the relaxed peak file, so stop
						break
				if (meanSummitChrom == currentSummit[0]) and \
					(abs(meanSummitPosition - currentSummit[1]) \
						<= summitDistCutoff):
					# The current peak is an ambiguous peak
					binaryMat[j,i] = -1
	meanSummitsFile.close()
	for regionFile in regionFileList:
		# Close the region files
		regionFile.close()
	for regionFileName in regionFileNameList:
		# Delete the filtered region files
		os.remove(regionFileName)
	np.savetxt(binaryMatAmigFileName, binaryMat, fmt='%d')

def makeSequenceInputArraysFromNarrowPeaksMultiTask(peaksFileName, optimalPeakFileListFileName, relaxedPeakFileListFileName, genomeFileName, \
			dataShape, mergePeaks=True, setAmbigPeaks=True, mergedSuffix="mergedPeaks", summitDistCutoff=50, summitIsOffset=1, createModelDir=False, \
			chroms=["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr2", "chr20", "chr22", \
				"chr3", "chr4", "chr5", "chr6", "chr7", "chr9", "chrX"], \
			bigWigFileNames=[], multiMode=False, streamData=False, dataFileName=""):
	# Convert each peak into a numpy array, where labelsFile has the labels for each task
	# ASSUMES THAT THIS IS FOR BINARY CLASSIFICATION TASKS
	# ASSUMES THAT peaksFile is sorted by CHROM., SUMMIT, START, END
	peaksFileNameElements = peaksFileName.split(".")
	peaksFileNamePrefix = ".".join(peaksFileNameElements[0:-2])
	mergedPeaksFileName = ".".join([peaksFileNamePrefix, mergedSuffix, "chroms.bed"])
	meanSummitsFileName = ".".join([peaksFileNamePrefix, mergedSuffix, "meanSummits.chroms.txt"])
	binaryMatFileName = ".".join([peaksFileNamePrefix, mergedSuffix, "chroms.binaryMat"])
	if mergePeaks:
		# The peaks have not been merged, so merge them
		from mergedPeaksSummitAndMakeBinaryMat import mergedPeaksSummitAndMakeBinaryMat
		mergedPeaksSummitAndMakeBinaryMat(peaksFileName, optimalPeakFileListFileName, mergedPeaksFileName, summitDistCutoff, meanSummitsFileName, \
			binaryMatFileName, summitIsOffset, chroms)
	binaryMatAmigFileName = ".".join([peaksFileNamePrefix, mergedSuffix, "ambigLabeled.chroms.binaryMat"])
	if setAmbigPeaks:
		# The ambiguous peaks have not been set to -1, so set them
		setAmbiguousPeaks(relaxedPeakFileListFileName, meanSummitsFileName, binaryMatFileName, binaryMatAmigFileName, chroms, \
			summitDistCutoff=summitDistCutoff)
	regionList = makeSummitPlusMinus(meanSummitsFileName, createOptimalBed=False, dataShape=dataShape, summitCol=1, startCol=None)
	modelDir = peaksFileNamePrefix + ".KerasModels"
	if createModelDir:
		# Create a new directory where the model parameters will be written
		os.mkdir(modelDir)
	halfWindowSize = dataShape[2]/2
	fastaFileName = ".".join([peaksFileNamePrefix, "plusMinus" + str(halfWindowSize) + "bp", "fa"])
	fasta = regionList.sequence(fi = genomeFileName, fo = fastaFileName)
	perBaseTracks = []
	if bigWigFileNames:
		# There are per-base tracks, so load them
		from bigWigFeaturizeOverBedNumpy import bigWigFeaturize
		perBaseTracks = bigWigFeaturize.new(bigWigFileNames, dataShape[2], intervals=[region for region in regionList], cache="cache")
	labels = np.repeat(np.loadtxt(binaryMatAmigFileName), 2, axis=0)
	print ("The total number of examples is: " + str(labels.shape[0]))
	channel1 = dataShape[0];
	channel2 = dataShape[1] + len(bigWigFileNames);
	channel3 = dataShape[2];
	allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.int8)
	if bigWigFileNames:
		# There are additional per-base tracks that might not be ints
		allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.float16)
	sampleCount = 0
	for record in SeqIO.parse(fastaFileName, "fasta"):
		# Iterate through the positive fastas and make a numpy array for each
		[allData, sampleCount] = addSequenceToArray(channel1, channel2, channel3, record, perBaseTracks, allData, sampleCount)
	assert (sampleCount == labels.shape[0])
	if multiMode:
		# Re-format the data for multi-moding
		allDataMultiModed = makeMultiModedData(allData, dataShape, len(perBaseTrackFileNames))
		if streamData:
			# Prepare the data for streaming instead of storing it
			saveDataToHDF5(dataFileName, allDataMultiModed, labels)
		return allDataMultiModed, labels, modelDir
	else:
		# Do not re-format the data for multi-moding
		if streamData:
			# Prepare the data for streaming instead of storing it
			saveDataToHDF5(dataFileName, allData, labels)
		return allData, labels, modelDir

def makeSequenceInputArrays(sequenceFileName, labelsFileName, dataShape, logLabels=False, standardizeLabels=False, perBaseTrackFileNames=[], multiMode=False, RC=True, maxFracNs = 1.0):
	# Convert each sequence into a numpy array
	# ASSUMES THAT EACH LABEL CORRESPONDS TO THE SEQUENCE ENTRY WITH THE SAME INDEX
	# ASSUMES THAT THE SEQUENCES ARE LISTS AND NOT IN FASTA FORMAT
	sequenceFile = open(sequenceFileName)
	perBaseTracks = loadPerBaseTracks(perBaseTrackFileNames)
	labelsNoRC = np.loadtxt(labelsFileName)
	labels = np.repeat(labelsNoRC, 2, axis = 0)
	if not RC:
		# Do not use the reverse complements
		labels = labelsNoRC
	if logLabels:
		# Log the values
		labels = labels + 0.0001 # Making sure no labels are 0 before log2ing
		assert(labels.all() > 0)
		labels = np.log2(labels)
	if standardizeLabels:
		# Standardize the values
		labels = (labels - self.meanTrain)/self.stdTrain
	channel1 = dataShape[0];
	channel2 = dataShape[1] + len(perBaseTracks);
	channel3 = dataShape[2];
	allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.int8)
	print ("The dimensions of the data are: " + str(allData.shape))
	if perBaseTracks:
		# There are additional per-base tracks that might not be ints
		allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.float16)
	sampleCount = 0
	totalNs = 0
	for sequence in sequenceFile:
		# Iterate through the fasta sequences and create the alphabets for the sequence and the reverse complement of each
		perBaseTracksMat = createPerBaseTracksMat(perBaseTracks, channel3, sampleCount, 2)
		sequenceArray, numNs = one_hot_encode(sequence.strip())
		fracNsSequence = float(numNs)/float(dataShape[2])
		if fracNsSequence > maxFracNs:
			# The current sequence has too high a percentage of N's, so do not include it
			labels = np.vstack((labels[0:sampleCount], labels[sampleCount + 2:len(labels)]))
			allData = allData[0:-2,:]
			continue
		totalNs = totalNs + numNs
		sequenceArrayReshape = np.reshape(np.vstack((sequenceArray, perBaseTracksMat)), (channel1, channel2, channel3))
		allData[sampleCount,:,:,:] = sequenceArrayReshape
		sampleCount = sampleCount + 1
		if RC:
			# Repeat for the reverse complement
			sequenceArrayRC = reverse_complement(sequenceArray)
			sequenceArrayReshapeRC = np.reshape(np.vstack((sequenceArrayRC, perBaseTracksMat)), (channel1, channel2, channel3))
			allData[sampleCount,:,:,:] = sequenceArrayReshapeRC
			sampleCount = sampleCount + 1
	print("The number of examples is: " + str(sampleCount))
	print("The number of labels is: " + str(labels.shape[0]))
	fracNs = float(totalNs)/float(dataShape[2] * sampleCount)
	if RC:
		# Multiply the fraction of N's by 2 because it does not include the reverse complements
		fracNs = fracNs * 2.0
        print("The fraction of Ns is: " + str(fracNs))
	assert (sampleCount == labels.shape[0])
	sequenceFile.close()
	if multiMode:
		# Re-format the data for multi-moding
		return makeMultiModedData(allData, dataShape, len(perBaseTrackFileNames)), labels
	return allData, labels
	
def makeSequenceInputArraysDiploid(sequenceMaternalFileName, sequencePaternalFileName, labelsFileName, dataShape, nucleotidesTogether=False, logLabels=False, standardizeLabels=False, perBaseTrackFileNames=[], multiMode=False, RC=True):
	# Convert each maternal, paternal sequence pair into a numpy array
	# ASSUMES THAT EACH LABEL CORRESPONDS TO THE SEQUENCE ENTRY WITH THE SAME INDEX
	# ASSUMES THAT THE SEQUENCES ARE LISTS AND NOT IN FASTA FORMAT
	sequenceMaternalFile = open(sequenceMaternalFileName)
	sequencePaternalFile = open(sequencePaternalFileName)
	perBaseTracks = loadPerBaseTracks(perBaseTrackFileNames)
	labelsNoRC = np.loadtxt(labelsFileName)
	labels = np.repeat(labelsNoRC, 4, axis = 0)
	if not RC:
		# Do not use the reverse complement
		labels = labelsNoRC
	if logLabels:
		# Log the values
		labels = labels + 0.0001 # Making sure no labels are 0 before log2ing
		assert(labels.all() > 0)
		labels = np.log2(labels)
	if standardizeLabels:
		# Standardize the values
		labels = (labels - self.meanTrain)/self.stdTrain
	channel1 = dataShape[0];
	channel2 = dataShape[1] + len(perBaseTracks);
	channel3 = dataShape[2];
	allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.int8);
	if perBaseTracks:
		# There are additional per-base tracks that might not be ints
		allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.float16);
	sampleCount = 0;
	for sequenceMaternal, sequencePaternal in izip(sequenceMaternalFile, sequencePaternalFile):
		# Iterate through the fasta sequences and create the alphabets in both directions for the sequence and the reverse complement of each
		perBaseTracksMat = createPerBaseTracksMat(perBaseTracks, channel3, sampleCount, 4)
		sequenceArrayMat, numNsMat = one_hot_encode(sequenceMaternal.strip())
		sequenceArrayPat, numNsPat = one_hot_encode(sequencePaternal.strip())
		sequenceArrayMatFirst = np.vstack((sequenceArrayMat, sequenceArrayPat, perBaseTracksMat))
		if nucleotidesTogether:
			# Put the As, Cs, Gs, and Ts together instead of putting each parent together
			sequenceArrayMatFirst = np.vstack((sequenceArrayMat[0,:], sequenceArrayPat[0,:], sequenceArrayMat[1,:], sequenceArrayPat[1,:], sequenceArrayMat[2,:], sequenceArrayPat[2,:], sequenceArrayMat[3,:], sequenceArrayPat[3,:], perBaseTracksMat))
		sequenceArrayReshapeMatFirst = np.reshape(sequenceArrayMatFirst, (channel1, channel2, channel3))	
		allData[sampleCount,:,:,:] = sequenceArrayReshapeMatFirst
		sampleCount = sampleCount + 1
		sequenceArrayPatFirst = np.vstack((sequenceArrayPat, sequenceArrayMat, perBaseTracksMat))
		if nucleotidesTogether:
			# Put the As, Cs, Gs, and Ts together instead of putting each parent together
			sequenceArrayPatFirst = np.vstack((sequenceArrayPat[0,:], sequenceArrayMat[0,:], sequenceArrayPat[1,:], sequenceArrayMat[1,:], sequenceArrayPat[2,:], sequenceArrayMat[2,:], sequenceArrayPat[3,:], sequenceArrayMat[3,:], perBaseTracksMat))
		sequenceArrayReshapePatFirst = np.reshape(sequenceArrayPatFirst, (channel1, channel2, channel3))
		allData[sampleCount,:,:,:] = sequenceArrayReshapePatFirst
		sampleCount = sampleCount + 1
		if RC:
			# Repeat for the reverse complement
			sequenceArrayRCMat = reverse_complement(sequenceArrayMat)
			sequenceArrayRCPat = reverse_complement(sequenceArrayPat)
			sequenceArrayRCMatFirst = np.vstack((sequenceArrayRCMat, sequenceArrayRCPat, perBaseTracksMat))
			if nucleotidesTogether:
				# Put the As, Cs, Gs, and Ts together instead of putting each parent together
				sequenceArrayRCMatFirst =\
					np.vstack((sequenceArrayRCMat[0,:], sequenceArrayRCPat[0,:], sequenceArrayRCMat[1,:], sequenceArrayRCPat[1,:], \
					sequenceArrayRCMat[2,:], sequenceArrayRCPat[2,:], sequenceArrayRCMat[3,:], sequenceArrayRCPat[3,:], perBaseTracksMat))
			sequenceArrayReshapeRCMatFirst = np.reshape(sequenceArrayRCMatFirst, (channel1, channel2, channel3))
			allData[sampleCount,:,:,:] = sequenceArrayReshapeRCMatFirst
			sampleCount = sampleCount + 1
			sequenceArrayRCPatFirst = np.vstack((sequenceArrayRCPat, sequenceArrayRCMat, perBaseTracksMat))
			if nucleotidesTogether:
				# Put the As, Cs, Gs, and Ts together instead of putting each parent together
				sequenceArrayRCPatFirst =\
					np.vstack((sequenceArrayRCPat[0,:], sequenceArrayRCMat[0,:], sequenceArrayRCPat[1,:], sequenceArrayRCMat[1,:], \
						sequenceArrayRCPat[2,:], sequenceArrayRCMat[2,:], sequenceArrayRCPat[3,:], sequenceArrayRCMat[3,:], perBaseTracksMat))
			sequenceArrayReshapeRCPatFirst = np.reshape(sequenceArrayRCPatFirst, (channel1, channel2, channel3))
			allData[sampleCount,:,:,:] = sequenceArrayReshapeRCPatFirst
			sampleCount = sampleCount + 1
	assert (sampleCount == labels.shape[0])
	sequenceMaternalFile.close()
	sequencePaternalFile.close()
	if multiMode:
		# Re-format the data for multi-moding
		return makeMultiModedData(allData, dataShape, len(perBaseTrackFileNames)), labels
	return allData, labels;
	
def makeSequenceInputArraysDiploidOneComb(sequenceMaternalFileName, sequencePaternalFileName, labelsFileName, dataShape, nucleotidesTogether=False, logLabels=False, standardizeLabels=False, perBaseTrackFileNames=[], multiMode=False):
	# Convert each maternal, paternal sequence pair into a numpy array
	# ASSUMES THAT EACH LABEL CORRESPONDS TO THE SEQUENCE ENTRY WITH THE SAME INDEX
	# ASSUMES THAT THE SEQUENCES ARE LISTS AND NOT IN FASTA FORMAT
	sequenceMaternalFile = open(sequenceMaternalFileName)
	sequencePaternalFile = open(sequencePaternalFileName)
	perBaseTracks = loadPerBaseTracks(perBaseTrackFileNames)
	labels = np.loadtxt(labelsFileName)
	if logLabels:
		# Log the values
		labels = labels + 0.0001 # Making sure no labels are 0 before log2ing
		assert(labels.all() > 0)
		labels = np.log2(labels)
	if standardizeLabels:
		# Standardize the values
		labels = (labels - self.meanTrain)/self.stdTrain
	channel1 = dataShape[0];
	channel2 = dataShape[1] + len(perBaseTracks);
	channel3 = dataShape[2];
	allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.int8);
	if perBaseTracks:
		# There are additional per-base tracks that might not be ints
		allData = np.zeros((len(labels), channel1, channel2, channel3), dtype=np.float16);
	sampleCount = 0;
	for sequenceMaternal, sequencePaternal in izip(sequenceMaternalFile, sequencePaternalFile):
		# Iterate through the fasta sequences and create the alphabets in both directions for the sequence and the reverse complement of each
		perBaseTracksMat = createPerBaseTracksMat(perBaseTracks, channel3, sampleCount, 4)
		sequenceArrayMat, numNsMat = one_hot_encode(sequenceMaternal.strip())
		sequenceArrayPat, numNsPat = one_hot_encode(sequencePaternal.strip())
		sequenceArrayMatFirst = np.vstack((sequenceArrayMat, sequenceArrayPat, perBaseTracksMat))
		if nucleotidesTogether:
			# Put the As, Cs, Gs, and Ts together instead of putting each parent together
			sequenceArrayMatFirst = np.vstack((sequenceArrayMat[0,:], sequenceArrayPat[0,:], sequenceArrayMat[1,:], sequenceArrayPat[1,:], sequenceArrayMat[2,:], sequenceArrayPat[2,:], sequenceArrayMat[3,:], sequenceArrayPat[3,:], perBaseTracksMat))
		sequenceArrayReshapeMatFirst = np.reshape(sequenceArrayMatFirst, (channel1, channel2, channel3))	
		allData[sampleCount,:,:,:] = sequenceArrayReshapeMatFirst
		sampleCount = sampleCount + 1
	assert (sampleCount == labels.shape[0])
	sequenceMaternalFile.close()
	sequencePaternalFile.close()
	return allData, labels;
	
def getDataInContext(inputMatrix, inputRegions, contextRegionsFileName=None, contextRegionsCol=None):
	# Filter the matrix to the TF context regions
	# inputMatrix - matrix with the input data, where each row is a feature
	# inputRegions - BedTool file with regions corresponding to the rows in the matrix
	# contextRegionsFileName - bed file with regions corresponding to the context that will be considered
	# contextRegionsCol - column corresponding to the context TF
	inputMatrixFilt = inputMatrix
	inputRegionsInContext = inputRegions
	nonZeroContextRows = None
	if contextRegionsFileName is not None:
		# There is a specific context for the input matrix
		print("Getting context regions!")
		assert (contextRegionsCol is None)
		assert(inputRegions is not None)
		contextRegions = bt.BedTool(contextRegionsFileName)
		inputRegionsInContext = inputRegions.intersect(contextRegions, c=True)
		inputRegionsInContextIndexes = []
		regionIndex = 0
		for iric in inputRegionsInContext:
			# Iterate through the input regions and make a list of the row numbers of those that are in the context regions
			if int(show_value(iric[3])) > 0:
				# The current region is in the context regions
				inputRegionsInContextIndexes.append(regionIndex)
			regionIndex = regionIndex + 1
		inputMatrixFilt = inputMatrix[np.array(inputRegionsInContextIndexes), :]
	if contextRegionsCol is not None:
		# Use a column to get the context regions
		print("Getting context regions!")
		assert (contextRegionsFileName is None)
		nonZeroContextRows = np.nonzero(inputMatrix[:, contextRegionsCol])[0]
		inputMatrixFilt = inputMatrix[nonZeroContextRows, :]
		inputRegionsInContext = inputRegions.at(nonZeroContextRows)
	bt.helpers.cleanup()
	return inputMatrixFilt, inputRegionsInContext, nonZeroContextRows
