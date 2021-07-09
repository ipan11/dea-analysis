# Run DEA analysis on the input dataframe and write outputs to csv
install.packages('MultiplierDEA')

library(MultiplierDEA)

# data1 <- read.csv("./datasets/medium_tute_dataset.csv", header=TRUE)
x <- read.csv("./datasets/x_normalised.csv", header=TRUE)
y <- read.csv("./datasets/y_normalised.csv", header=TRUE)

# Variable returns to scale model and output orientation
result <- DeaMultiplierModel(x=x,y=y,"vrs", "output")
# Examine the efficiency score for DMUs
print(result$Efficiency)

resultMerged <- do.call("cbind", list(result$InputValues, result$OutputValues, result$Efficiency, result$vx, result$uy))

write.csv(resultMerged, "./results/dea_result.csv")




