import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

CTCFsOverlappVal = np.loadtxt("/srv/scratch/imk1/TFBindingPredictionProject/CTCF-sData/biotin/CTCFPeak/peak/spp/idr/optimal_set/CTCFPeak_rep1-pr.IDR0.05.filt_inCTCFs_bestCoreDownstreampVal.txt")
nonCTCFsOverlappVal = np.loadtxt("/srv/scratch/imk1/TFBindingPredictionProject/CTCF-sData/biotin/CTCFPeak/peak/spp/idr/optimal_set/CTCFPeak_rep1-pr.IDR0.05.filt_notInCTCFs_bestCoreDownstreampVal.txt")
plot = sns.violinplot(data=[0.0-np.log10(nonCTCFsOverlappVal), 0.0-np.log10(CTCFsOverlappVal)])
fig = plot.get_figure()
fig.savefig("/srv/scratch/imk1/TFBindingPredictionProject/CTCF-sData/biotin/CTCFPeak/peak/spp/idr/optimal_set/CTCFPeak_rep1-pr.IDR0.05.filt_inVsNotInCTCFs_bestCoreDownstreampVal.svg")

plt.clf()
EncodeOverlappVal = np.loadtxt("/srv/scratch/imk1/TFBindingPredictionProject/CTCF-sData/biotin/CTCFPeak/peak/spp/idr/optimal_set/CTCFPeak_rep1-pr.IDR0.05.filt_inEncodeHela_bestCoreDownstreampVal.txt")
nonEncodeOverlappVal = np.loadtxt("/srv/scratch/imk1/TFBindingPredictionProject/CTCF-sData/biotin/CTCFPeak/peak/spp/idr/optimal_set/CTCFPeak_rep1-pr.IDR0.05.filt_notInEncodeHela_bestCoreDownstreampVal.txt")
EncodePlot = sns.violinplot(data=[0.0-np.log10(nonEncodeOverlappVal), 0.0-np.log10(EncodeOverlappVal)])
EncodeFig = EncodePlot.get_figure()
EncodeFig.savefig("/srv/scratch/imk1/TFBindingPredictionProject/CTCF-sData/biotin/CTCFPeak/peak/spp/idr/optimal_set/CTCFPeak_rep1-pr.IDR0.05.filt_inVsNotInEncode_bestCoreDownstreampVal.svg")
