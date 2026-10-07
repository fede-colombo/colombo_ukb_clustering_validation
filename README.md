# Mapping reproducible brain-based depression stratification across clinical, cognitive, and neurotransmitter dimensions

This repository contains code and data in support of "Mapping reproducible brain-based depression stratification across clinical, cognitive, and neurotransmitter dimensions". 
All code was written in Python and R. Below I describe the contents of this repository.

## code
The code folder contains all the scripts to run the analyses. A description of each subfolder follows:
- **01_clustering**: this folder contains all the scripts used to run NeuReval in the UKB training set and validate the optimal clustering solution in the UKB hold-out test set and in the HSR external cohort. More details about NeuReval can be found in the dedicated [GitHub repo](https://github.com/fede-colombo/NeuReval).
- **02_neurosynth**: this folder includes script to parcellate Neurosynth data in 62 regions based on the DKT atlas and run Partial Least Squares analyses to map clusters' cortical profiles onto the Neurosynth functional activation maps. Codes were adapted from [hansen_receptors](https://github.com/netneurolab/hansen_receptors) and [
ipn-summer-school](https://github.com/netneurolab/ipn-summer-school/tree/main/lectures/2021-07-02/13-15).
- **03_receptors**: this folder contains the scripts to parcellate 19 PET-derived maps of neurotransmitter receptors and transporters (DKT atlas) and run multilinear regression models with dominance analyses to predict clusters' cortical profiles. Codes were adapted from [hansen_receptors](https://github.com/netneurolab/hansen_receptors).
- **utils**: this folder includes additonal scripts to perform clusters' comparisons for input or outcome variables, run sigclust to assess the statistical signiifcance of clustering solutions, as well as scripts for matching (greedy matching and propensity score full matching).

## data
The data folder contains data files used for the analyses. If you use this data in your own analyses, please cite the associated papers.
- **CT_cohend_*.csv** files are the regional Cohen's d values for each dataset (UKB training, UKB test, and HSR validation cohort)
- **dkt_parcellated_neurosynth_123.csv** are the parcellated Neurosynth association maps for 123 terms from the Cognitive Atlas. The original maps are available [here](https://github.com/netneurolab/ipn-summer-school/tree/main/lectures/2021-07-02/13-15).
- **dkt_receptor_data_matrix.csv** are the parcellated PET maps of neurotrasmitters receptors and transporters. The original maps are available at [hansen_receptors](https://github.com/netneurolab/hansen_receptors).
- The **atlas** folder contains the DKT coordinates and annot files used to parcellate Neurosynth and PET maps.
