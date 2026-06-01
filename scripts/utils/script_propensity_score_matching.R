######################################
### PROPENSITY SCORE FULL MATCHING ###
######################################

# Install packages if needed
# install.packages("MatchIt")
# install.packages("cobalt")

library(MatchIt)
library(cobalt)

# Set path to data to match
df <- read.csv("path/to/data_to_match.csv", sep=';')

# Ensure correct variable types
df$CASE <- as.numeric(df$CASE)
df$SEX  <- as.factor(df$SEX)
df$AGE  <- as.numeric(df$AGE)
df$BMI  <- as.numeric(df$BMI)

# Simple mean imputation
df$BMI_imputed <- df$BMI
df$BMI_imputed[is.na(df$BMI_imputed)] <- mean(df$BMI, na.rm = TRUE)

# Perform matching: full matching, keep all the cases
match_out <- matchit(
  CASE ~ AGE + BMI_imputed,
  data = df,
  method = "full",
  exact = ~ SEX,
  distance = "glm"
)

# Summary of matching
summary(match_out)

# Extract matched dataset
matched_data <- match.data(match_out)

# Check covariate balance visually
love.plot(match_out, binary = "std")

# View first rows of matched data
head(matched_data)

# Save matched data
write.csv(matched_data, 'path/to/data_matched_propensity_score_age_sex_bmi.csv')
