###########################################
## Get atlas coordinates from brainGraph ##
###########################################

# Load libraries
library(brainGraph)
library(readxl)

# Get your atlas coordinates
# For a complete list of the atlases in brainGraph, see https://www.rdocumentation.org/packages/brainGraph/versions/3.1.1/topics/Brain%20Atlases
coord <- dkt

# Save coordinates
write.csv(coord, 'path/to/dkt_coord.csv')
