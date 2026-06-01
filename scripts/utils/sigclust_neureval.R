library(sigclust)
library(ggplot2)
library(readxl)

set.seed(42)

# IMPORTANT:
# 1. Data must first be normalized and corrected for the same covariates used in the analyses
# 2. Labels must be 1 and 2 (NOT 0 and 1)
data <- read.csv("path/to/data_UKB_55_training_DKT_with_labels_corrected.csv", sep=';')
nreps_sim = 10000
nreps_sigclust_repeat = 10000
data$Labels[data$Labels == 1] <- 2
data$Labels[data$Labels == 0] <- 1
data$Labels = factor(data$Labels, levels=c("1","2"))

# OBSERVATIONS:
# Results vary greatly depending on the icovest parameter --> defines how the eigenvalues of the covariance matrix are estimated:
# 1: Soft Thresholding: recommended for high-dimensional data (sparse covariance matrix estimation methods), returns a more truthful p-value compared to sample and hard-thresholding
# 2: Sample: recommended for low-dimensional data, uses the sample covariance (very conservative approach)
# 3: Hard thresholding: uses the background noise thresholded estimate (very anti-conservative approach)
# By default sigclust is based on soft-thresholding -> The difference between the soft thresholding method (13) and the hard thresholding
# estimators (3) is that the large eigenvalues are subtracted by a constant τ . Since the hard thresholding approach
# tends to overestimate some eigenvalues, the soft thresholding approach adds some constraints to reduce the largest eigenvalues.
cluster_sig = sigclust(data[,c(2:63)], nsim=nreps_sim, nrep=nreps_sigclust_repeat, labflag=1, label=data[,"Labels"], icovest=1)
plot(cluster_sig)

# p_val_train: sum of how many cluster indices generated on the real data are less than or equal to those obtained from the simulated data, divided by the number of repetitions
p_val_train = (sum(cluster_sig@simcindex<=cluster_sig@xcindex)+1)/(nreps_sim+1)
print(p_val_train)

