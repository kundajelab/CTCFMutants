library('DESeq2')
directory<-'/srv/scratch/imk1/TFBindingPredictionProject/MouseMutantData/'
sampleMat = matrix(nrow = 5, ncol=2)
colnames(sampleMat) <- c("", "Mutant") # 2nd column will be used for the design
samples <- read.table(file.path(directory, "indivRep_allPeaks-readsPerSample.txt"), header=FALSE)
colnames(samples) <- c("R339W", "R339W", "R339W", "ZF8", "ZF8", "ZF8", "ZF10", "ZF10", "ZF10", "ZF11", "ZF11", "ZF11", "ZF2", "ZF2", "ZF2", "ZF3", 
						"ZF3", "ZF3", "ZF4", "ZF4", "ZF4", "ZF5", "ZF5", "ZF5", "ZF6", "ZF6", "ZF6", "ZF7", "ZF7", "ZF7", "ZF9", "ZF9", "ZF9", "WT", 
						"WT", "ZF1", "ZF1", "ZF1")
sampleMat[1,1] = "WT-1"
sampleMat[1,2] = "WT"
sampleMat[2,1] = "WT-2"
sampleMat[2,2] = "WT"
sampleMat[3,1] = "ZF1-1"
sampleMat[3,2] = "ZF1"
sampleMat[4,1] = "ZF1-2"
sampleMat[4,2] = "ZF1"
sampleMat[5,1] = "ZF1-3"
sampleMat[5,2] = "ZF1"
sampleMatFrame = data.frame(sampleMat)
samplesTask = samples[c(34:38)]
dds <- DESeqDataSetFromMatrix(countData = samplesTask, colData = sampleMatFrame, design = ~Mutant)
dds <- DESeq(dds)
res<-results(dds)
write.table(as.data.frame(res), file="/srv/scratch/imk1/TFBindingPredictionProject/MouseMutantData/WTvsZF1.csv", quote=FALSE, 
	sep="\t", row.names=FALSE, col.names=FALSE)
