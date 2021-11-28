import sys
import argparse

def parseArgument():
	# Parse the input
	parser=argparse.ArgumentParser(description=\
			"Get the motif hits from a FIMO txt output file and record them in a bed file")
	parser.add_argument("--fimoFileName", required=True,\
			help='File from FIMO')
	parser.add_argument("--nameCol", type=int, required=False, default=1,\
			help='Column of the fimo.txt file with the motif name')
	parser.add_argument("--scoreCol", type=int, required=False, default=6,\
			help='Column of the fimo.txt file with the motif match score')
	parser.add_argument("--pValCutoff", type=float, required=False, default=0.05,\
			help='p-value cutoff for filtering FIMO file')
	parser.add_argument("--qValCutoff", type=float, required=False, default=1.01,\
			help='q-value cutoff for filtering FIMO file')
	parser.add_argument("--scoreCutoff", type=float, required=False, default=0,\
			help='score cutoff for filtering FIMO file')
	parser.add_argument("--outputFileName", required=True,\
			help='bed file where the motif hits will be written')
	parser.add_argument("--posTab", action='store_true', required=False, help="The position is tab-deliminted, not in format chr:start-end")
	options = parser.parse_args()
	return options

def convertFIMOToMotifHitBed(options):
	# Get the motif hits from a FIMO txt output file and record them in a bed file
	fimoFile = open(options.fimoFileName)
	fimoFile.readline() # Remove the header
	outputFile = open(options.outputFileName, 'w+')
	for line in fimoFile:
		# Iterate through the lines of the FIMO file and record the motif location for each
                if line[0].startswith("#"):
                    # Skip the current line
                    continue
                lineElements = line.strip().split("\t")
                if ((float(lineElements[7]) >= options.pValCutoff) or (float(lineElements[8]) >= options.qValCutoff)) or (float(lineElements[6]) < options.scoreCutoff):
		    # The hit is not sufficiently significant, so skip it
                    continue
                chrom = ""
                start = 0
                end = 0
                if not options.posTab:
		    # The positions are in format chr:start-end, strand indicates the strand of the motif, but the motif always starts upstream
                    locationElements = lineElements[2].split(":")
                    chrom = locationElements[0]
                    positionElements = [int(pos) for pos in locationElements[1].split("-")]
                    start = positionElements[0] + int(lineElements[3]) - 1
                    end = positionElements[0] + int(lineElements[4])
                else:
		    # The positions are tab-delimited and are the motif hit instead of the full region
                    chrom = lineElements[2]
                    start = int(lineElements[3])
                    end = int(lineElements[4])
                outputFile.write("\t".join([chrom, str(start), str(end), lineElements[options.nameCol], lineElements[options.scoreCol], lineElements[5]]) + "\n")
	fimoFile.close()
	outputFile.close()

if __name__ == "__main__":
	options = parseArgument()
	convertFIMOToMotifHitBed(options)
