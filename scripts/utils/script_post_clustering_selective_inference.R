# install.packages("VALIDICLUST")
# install.packages("cluster")
library(cluster)
library(VALIDICLUST)
library(reticulate)
library(readxl)

#install_miniconda()          # first time only
use_condaenv("r-reticulate", required = TRUE)
py_config()

# To make sure that the conda environment and related packages are the same as in NeuReval, you have to install it first
py_install("neureval==1.1.1", pip = TRUE)

# Import other required packages
pickle <- import("pickle")
builtins <- import_builtins()

### SETUP ###

# Set path to your data
# The data passed to the function should already be standardized and confound-corrected as in NeuReval
data_path = "/path/to/project/results/comparisons/data_with_labels.csv"

# Define output path
out_path = "/path/to/project/results/post_clustering_inference"

# Define the path to the findbestclust object
findbestclust_path = "/path/to/project/results/clustering/findbestclust_model.pkl"

#####################################################################################

# Load data
data <- read.csv(data_path, sep=';')
# Define data to use (transform them into a matrix)
#data <- data[data$train_test == 'TR',]
data_to_use <- as.matrix(data[,2:63])
# Define the column Labels
Labels <- as.integer(data[,64]) + 1
  
# Load the findbestclust model
f <- builtins$open(findbestclust_path, "rb")
findbestclust_model <- pickle$load(f)
f$close()

# For the findbestclust model, extract the optimized KMeans algorithm
  best_clust = findbestclust_model$clust_method

# Define the KMeans function using the optimized KMeans
best_clust_func <- function(x){
  best_clust$fit(x)
  as.integer(best_clust$labels_)+1}

# Get labels predicted from KMeans
predicted <- best_clust_func(data_to_use)

# Check if there is consistency between the predicted labels and the original labels
# Note: there will probably be a small mismatch, since the original labels derive from the cross-validation procedure built into NeuReval
concordance_table <- table(predicted, Labels)
print(concordance_table)

# Run the selective inference test
# Note: the selective inference test is performed on one single feature, so you have to loop over all the features and store all p-values
n_features <- dim(data_to_use)[2]

p_values <- rep(NA_real_, n_features)
stat_g_values <- rep(NA_real_, n_features)

for (i in 1:n_features) {
  res <- test_selective_inference(data_to_use, k1 = 1, k2 = 2, g = i,
                                  cl_fun = best_clust_func, ndraws = 5000,
                                  cl = predicted)
  p_values[i] <- res$pval
  stat_g_values[i]   <- res$stat_g
  cat("Feature", i, "/", n_features, "- pval:", round(res$pval, 4), "\n")
}

# Adjust for FDR multiple comparisons
p_adj <- p.adjust(p_values, method = "BH")   # or "bonferroni"


# Save results
results_df <- data.frame(
  feature   = colnames(data_to_use),
  pval      = p_values,
  stat_g    = stat_g_values,
  pval_adj  = p_adj
)

if (!dir.exists(out_path)) dir.create(out_path, recursive = TRUE)
write.csv(results_df, file.path(out_path, "selective_inference_results.csv"), row.names = FALSE)

print(results_df)
