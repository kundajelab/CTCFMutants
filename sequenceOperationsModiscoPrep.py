import sys
import os
import subprocess
import pybedtools as bt
import numpy as np
from Bio import SeqIO

def show_value(s):
        """
        Convert unicode to str under Python 2;
        all other values pass through unchanged
        """
        if sys.version_info.major == 2:
                if isinstance(s, unicode):
                        return str(s)
        return s

def createBedToolForFilteredList(regionList, createBedFilt, chroms, bedFiltFileName=None):
        # Get the BedTool for a filtered set of peaks
        if createBedFilt:
                # Create a bed file for a filtered set of peaks
                regionListFilt = bt.BedTool([region for region in regionList if region[0] in chroms])
                regionListFilt.saveas(bedFiltFileName)
        if bedFiltFileName:
                # Save the region list to a file
                regionListFilt = bt.BedTool(bedFiltFileName)
        bt.helpers.cleanup()
        return regionListFilt

def makeSummitPlusMinus(optimalPeakFileName, createOptimalBed=False, dataShape=(1,4,1000), summitCol=9, startCol=1,
        bedFilegzip=False, chroms=None, maxPeakLength=None):
        # Create a bed file that is the summit +/- half the window bp for each peak summit
        optimalPeakFileNameElements = optimalPeakFileName.split(".")
        optimalPeakFileNamePrefix = ".".join(optimalPeakFileNameElements[0:-2])
        optimalBedFileName = optimalPeakFileNamePrefix + ".bed"
        if createOptimalBed:
                # Create a bed file for the optimal peaks
                os.system(" ".join(["zcat", optimalPeakFileName, \
                        "| grep -v chrUn | grep -v random | grep chr | sort -k1,1 -k2,2n -k3,3n -k" + str(summitCol + 1) +"," + str(summitCol+ 1) + \
                        "n >", optimalBedFileName]))
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
                        bt.BedTool([bt.Interval(r[0], int(int(r[startCol]) + int(r[summitCol]) - halfWindowSize), int(int(r[startCol])\
                                + int(r[summitCol]) + halfWindowSize))\
                                for r in optimalRegionListFilt])
        else:
                # The start postions are not in the file (summit list and not narrowPeak file)
                summitPlusMinus =\
                        bt.BedTool([bt.Interval(r[0], int(int(r[summitCol]) - halfWindowSize), \
                                int(int(r[summitCol]) + halfWindowSize))\
                                for r in optimalRegionListFilt])
        if chroms:
                # Filter the summitsPlusMinus list based on the chromosomes
                summitPlusMinus = createBedToolForFilteredList(summitPlusMinus, True, chroms)
        bt.helpers.cleanup()
        return summitPlusMinus

def defineInterval(r, halfWindowSize, summitPresent, windowSizeOdd=False):
        # Create an interval that the CNN can take from a region from a bed file
        chrom = show_value(r[0])
        if summitPresent:
                # Convert the region to a summit-centered interval
                start = int(int(show_value(r[1])) + int(show_value(r[9])) - halfWindowSize)
                if windowSizeOdd:
                        # Subtract 1 from the start
                        start = start - 1
                end = int(int(show_value(r[1])) + int(show_value(r[9])) + halfWindowSize)
                return [chrom, start, end]
        else:
                # Use the centers of the peaks instead of summits
                start = int(int(show_value(r[1])) + int(round((float(show_value(r[2])) - float(show_value(r[1])))/2.0)) - halfWindowSize)
                if windowSizeOdd:
                        # Subtract 1 from the start
                        start = start - 1
                end = int(int(show_value(r[1])) + int(round((float(show_value(r[2])) - float(show_value(r[1])))/2.0)) + halfWindowSize)
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
                                print ("Chromosome " + chrom + " is not in the list of chromosomes")
                                continue
                        if end > chromSizesDict[chrom] - chromEdgeDistLimit:
                                # Do not use the current region because it is too close to the end of the chromosome
                                print ("End greater than chromosome length - chromEdgeDistLimit for region: " + str(r))
                                continue
                if (maxPeakLength != None) and (int(round(float(show_value(r[2])) - float(show_value(r[1])))) > maxPeakLength):
                        # The current region is too log, so skip it
                        continue
                regionListFiltList.append(r)
                #intervalList.append(bt.Interval(chrom, start, end, show_value(r[4])))
                intervalList.append(bt.Interval(chrom, start, end, show_value(r[3])))
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

def createPositiveSetFromNarrowPeaks(optimalPeakFileName, genomeFileName, dataShape, createOptimalBed=False, createOptimalBedFilt=True, \
        maxPeakLength=None, chroms=None, chromSizesFileName=None, chromEdgeDistLimit=0):
        # Create the positive set for the deep learning model
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

def createPerBaseTracksMat(perBaseTracks, width, sampleCount, divisor):
        # Create a matrix with the per base tracks for the current sample
        # ASSUMES THAT sampleCount IS A MULTIPLE OF divisor
        perBaseTracksIndex = sampleCount / divisor
        perBaseTracksMat = np.empty((0, width))
        for pbt in perBaseTracks:
                # Iterate through the per base
                perBaseTracksMat = np.vstack((perBaseTracksMat, pbt[perBaseTracksIndex, :]))
        return perBaseTracksMat

def oneHotEncode(sequence):
        encodedSequence = np.zeros((4, len(sequence)), dtype=np.int8)
        sequenceDict = {}
        sequenceDict["A"] = np.array([1, 0, 0, 0])
        sequenceDict["a"] = np.array([1, 0, 0, 0])
        sequenceDict["C"] = np.array([0, 1, 0, 0])
        sequenceDict["c"] = np.array([0, 1, 0, 0])
        sequenceDict["G"] = np.array([0, 0, 1, 0])
        sequenceDict["g"] = np.array([0, 0, 1, 0])
        sequenceDict["T"] = np.array([0, 0, 0, 1])
        sequenceDict["t"] = np.array([0, 0, 0, 1])
        sequenceDict["N"] = np.array([0, 0, 0, 0])
        sequenceDict["n"] = np.array([0, 0, 0, 0])
        # These are all 0's even though they should ideally have 2 indices with 0.5's because storing ints requires less space than storing floats
        sequenceDict["R"] = np.array([0, 0, 0, 0])
        sequenceDict["r"] = np.array([0, 0, 0, 0])
        sequenceDict["Y"] = np.array([0, 0, 0, 0])
        sequenceDict["y"] = np.array([0, 0, 0, 0])
        sequenceDict["M"] = np.array([0, 0, 0, 0])
        sequenceDict["m"] = np.array([0, 0, 0, 0])
        sequenceDict["K"] = np.array([0, 0, 0, 0])
        sequenceDict["k"] = np.array([0, 0, 0, 0])
        sequenceDict["W"] = np.array([0, 0, 0, 0])
        sequenceDict["w"] = np.array([0, 0, 0, 0])
        sequenceDict["S"] = np.array([0, 0, 0, 0])
        sequenceDict["s"] = np.array([0, 0, 0, 0])
        for i in range(len(sequence)):
                # Iterate through the bases in the sequence and record each
                encodedSequence[:,i] = sequenceDict[sequence[i]]
        numNs = len(sequence) - np.sum(encodedSequence)
        return encodedSequence, numNs

def reverse_complement(encoded_sequences):
        # Because the encoding is A, C, G, T in that order, can just reverse each sequence along both axes.
        return encoded_sequences[..., ::-1, ::-1]

def addSequenceToArray(channel1, channel2, channel3, sequenceRecord, perBaseTracks, allData, sampleCount, perBaseTracksDivisor=2):
        # Add a sequence and its reverse complement to a numpy array
        perBaseTracksMat = createPerBaseTracksMat(perBaseTracks, channel3, sampleCount, perBaseTracksDivisor)
        sequenceArray, numNs = oneHotEncode(str(sequenceRecord.seq).strip())
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
        perBaseTracksMat = createPerBaseTracksMat(perBaseTracks, channel3, sampleCount, 2)
        sequenceArray, numNs = oneHotEncode(str(sequenceRecord.seq).strip())
        sequenceArrayReshape = np.reshape(np.vstack((sequenceArray, perBaseTracksMat)), (channel1, channel2, channel3))
        allData[sampleCount,:,:,:] = sequenceArrayReshape
        sampleCount = sampleCount + 1
        return [allData, sampleCount]

def makePositiveSequenceInputArraysFromFasta(positiveFastaFileName, dataShape=(1,4,1000), labels=np.array([]), RC=False):
        # Convert each peak into a numpy array, where each peak is inputted in fasta format
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
                        # The current row has a large fold-change in the wild-type direction and a small p-value, so add the region to the negative set
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
                        positiveFastaFileName, negativeFastaFileName, dataShape, bigWigFileNames=bigWigFileNames, multiMode=multiMode, \
                        streamData=streamData, dataFileName=dataFileName, RC=RC, removeFastas=removeFastas)
        positiveIndicesList.extend(negativeIndicesList)
        if swapLabels:
                # Swap the positive and negative labels
                labels = 1 - labels
        bt.helpers.cleanup()
        return allData, labels, positiveFastaFileName, negativeFastaFileName, modelDir, np.array(positiveIndicesList)
