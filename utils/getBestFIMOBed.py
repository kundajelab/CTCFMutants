def getFIMOLocations(FIMOFileName, newFormat):
        # Get the regions in a FIMO file
        # Returns an dictionary for each chromosome where each entry has an array of regions whose entries are (chromosome, start, end)
        FIMOFile = open(FIMOFileName)
        FIMOLocations = {}
        for line in FIMOFile:
	    # Iterate through the lines of the FIMO file and record the locations
	    if line[0] == "#":
		# The current line is a header, so skip it
		continue
	    lineElements = line.split()
	    currentLocation = ("", 0, 0, 1)
	    if newFormat == 0:
		# The file is in the old format; strand indicates the strand of the motif, but the motif always starts upstream
		locationElements = lineElements[1].split(":")
		positionElements = locationElements[1].split("-")
		currentLocation =\
		    (locationElements[0], int(positionElements[0]) + int(lineElements[2]), int(positionElements[0]) + int(lineElements[3]), \
		        float(lineElements[6]))
	    else:
		# The file is in the new format
		currentLocation =\
		    (lineElements[1], int(lineElements[2]), int(lineElements[3]), float(lineElements[6]))
	    if currentLocation[0] not in FIMOLocations:
		# Add the current chromosome to the chromosome dictionary
		FIMOLocations[currentLocation[0]] = []
	    FIMOLocations[currentLocation[0]].append(currentLocation)
        FIMOFile.close()
        return FIMOLocations

def getBestFIMOBed(FIMOFileName, bedFileName, outputFileName, newFormat):
        # Get the best FIMO hit in a bed file
        # Outputs a list that is number of bed regions long, where each value is the p-value of the best FIMO hit in the region
        # ASSUMES THAT THE FIMO FILE IS SORTED BY P-VALUE FROM LOWEST TO HIGHEST
        FIMOLocations = getFIMOLocations(FIMOFileName, newFormat)
        print("FIMO locations dictionary has been made!")
        bedFile = open(bedFileName)
        outputFile = open(outputFileName, 'w+')
        for line in bedFile:
                # For each line of the bed file, find the FIMO intersection with the lowest p-value
                lineElements = line.split("\t")
                chrom = lineElements[0]
                start = int(lineElements[1])
                end = int(lineElements[2])
                bestFIMOpVal = 1
                if chrom in FIMOLocations.keys():
		    # There is a FIMO hit on the current chromosome
		    for FIMOLoc in FIMOLocations[chrom]:
			# Iterate through FIMO locations to determine which locations intersect the BED location
			if (((start <= FIMOLoc[1]) and (end >= FIMOLoc[1])) or ((end >= FIMOLoc[2]) and (start <= FIMOLoc[2]))) or \
                            ((start >= FIMOLoc[1]) and (end <= FIMOLoc[2])):
			    # The regions overlap, so this must be the best match
			    assert (FIMOLoc[3] <= bestFIMOpVal)
			    bestFIMOpVal = FIMOLoc[3]
			    break
                outputFile.write(str(bestFIMOpVal))
                outputFile.write("\n")
        bedFile.close()
        outputFile.close()

if __name__=="__main__":
   import sys
   FIMOFileName = sys.argv[1] 
   bedFileName = sys.argv[2]
   outputFileName = sys.argv[3]
   newFormat = 0
   if len(sys.argv) > 4:
	   newFormat = int(sys.argv[4])
   getBestFIMOBed(FIMOFileName, bedFileName, outputFileName, newFormat)
