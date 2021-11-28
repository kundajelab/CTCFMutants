import sys
import argparse
import numpy as np


def parseArgument():
	# Parse the input
	parser=argparse.ArgumentParser(description=\
			"Get the positions with the top scores and their scores")
	parser.add_argument("--regionsFileName", required=True,\
			help='List of regions file, assumed to be a bed file if --chromCol, --startCol, and --endCol are not specified')
	parser.add_argument("--scoresFileName", required=True,\
			help='List of scores file')
	parser.add_argument("--outputFileName", required=True,\
			help='File where regions, their top positions, and the scores of those positions will be recorded')
	parser.add_argument("--numScoresPerRegion", type=int, default=2, required=False,\
			help='Number of per-base score lists for each region')
	parser.add_argument("--trackName", required=True,\
			help='Name of the bedGraph track')
	parser.add_argument("--chromCol", type=int, default=0, required=False,\
			help='Column of the regions file with the chromosome')
	parser.add_argument("--startCol", type=int, default=1, required=False,\
			help='Column of the regions file with the start')
	parser.add_argument("--endCol", type=int, default=2, required=False,\
			help='Column of the regions file with the end')
	parser.add_argument("--excludeRC", action='store_true', required=False,\
			help='Exclude the reverse complements')
	parser.add_argument("--onlyRC", action='store_true', required=False,\
			help='Use only the reverse complements')
	options = parser.parse_args()
	return options


def makeBedGraphFromPositionScores(options):
	# Get the positions with the largest scores
	regionsFile = open(options.regionsFileName)
	scores = np.loadtxt(options.scoresFileName)
	outputFile = open(options.outputFileName, 'w+')
	# Write the header to the output file
	outputFile.write("track type=bedGraph name=\"" + options.trackName + "\"\n")
	scoresIndex = 0
	for line in regionsFile:
		# Iterate through the regions and get the top-scoring positions for each
		lineElements = line.strip().split("\t")
		peakChrom = lineElements[options.chromCol]
		peakStart = int(lineElements[options.startCol])
		peakEnd = int(lineElements[options.endCol])
		for i in range(options.numScoresPerRegion):
			# Iterate through the score list for the current region
			# (there could be 2 scores, 1 for the seq. and 1 for the RC)
			if options.excludeRC and (i % 2 != 0):
				# Skip the reverse complement
				scoresIndex = scoresIndex + 1
				continue
			if options.onlyRC and (i % 2 == 0):
				# Skip the reverse complement
				scoresIndex = scoresIndex + 1
				continue
			for j in range(scores.shape[1]):
				# Iterate through the positions and record all of them
				posStart = peakStart + j
				posEnd = peakStart + j + 1
				s = scores[scoresIndex, j if i % 2 == 0 else scores.shape[1] - j - 1] # ASSUMES SEQ FOLLOWED BY RC
				locationString = "{0}\t{1}\t{2}".format(peakChrom, str(posStart), str(posEnd))
				outputString = "{0}\t{1}\n".format(locationString, str(s))
				outputFile.write(outputString)
			scoresIndex = scoresIndex + 1
	regionsFile.close()
	outputFile.close()


if __name__ == "__main__":
	options = parseArgument()
	makeBedGraphFromPositionScores(options)
