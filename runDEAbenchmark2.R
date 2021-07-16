# Run DEA analysis on the input dataframe and write outputs to csv
# Use Benchmarking package
# install.packages('Benchmarking')

library(Benchmarking)

x <- read.csv("./datasets/x_normalised.csv", header=TRUE)
y <- read.csv("./datasets/y_normalised.csv", header=TRUE)

# Variable returns to scale model and output orientation
result <- dea(X=x, Y=y, RTS="vrs", ORIENTATION="out")

# Write the efficiency score for DMUs
write.csv(list(result$eff), "./results/R_Benchmarking_lib_DEA_outputs.csv")

# Benchmarking library returns ux and vy - to check


