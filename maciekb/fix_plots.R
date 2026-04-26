# maciekb/fix_plots.R
data <- read.csv("for_participants/data/players_quarters_final.csv", stringsAsFactors = FALSE)

# We use Base R to ensure compatibility
png("maciekb/plot_shots_target.png", width=800, height=500)

# Create a jittered plot to show the distribution of discrete counts
# Scorers are 1, Non-scorers are 0
scorers <- data$last15_shots_on_target[data$scored_after == 1]
non_scorers <- data$last15_shots_on_target[data$scored_after == 0]

# Add a bit of random noise (jitter) to the X and Y coordinates 
# so we can see the density of the points
plot(jitter(data$scored_after, factor=0.5), 
     jitter(data$last15_shots_on_target, factor=0.5),
     pch=16, col=rgb(0.1, 0.4, 0.8, 0.3),
     xaxt="n", xlab="Will Score Later? (0=No, 1=Yes)", 
     ylab="Shots on Target (Last 15m)",
     main="Shot Accuracy Density: Scorers vs Non-Scorers")
axis(1, at=c(0, 1))

# Add mean points in red to show the trend
points(c(0, 1), c(mean(non_scorers), mean(scorers)), 
       col="red", pch=18, cex=2)
legend("topright", legend="Mean Value", col="red", pch=18)

dev.off()
cat("Fixed Shots on Target plot saved to maciekb/plot_shots_target.png\n")
