# maciekb/clean_heatmap.R
data <- read.csv("for_participants/data/players_quarters_final.csv", stringsAsFactors = FALSE)

# Select performance columns
numeric_cols <- data[, sapply(data, is.numeric)]
performance_cols <- numeric_cols[, !names(numeric_cols) %in% c("player_appearance_id", "player_id", "fixture_id", "jersey_number")]

# Calculate correlation matrix
cor_matrix <- cor(performance_cols, use = "complete.obs")

# Create a clean heatmap WITHOUT dendrograms (the "branches")
png("maciekb/plot_correlation_heatmap.png", width = 800, height = 800)

# Set up plot margins
par(mar=c(12, 12, 4, 2))

# image() creates the clean grid
# We transpose the matrix so it looks correct (X/Y orientation)
image(1:ncol(cor_matrix), 1:nrow(cor_matrix), t(cor_matrix), 
      col = hcl.colors(12, "RdBu", rev = TRUE), 
      axes = FALSE, xlab="", ylab="", 
      main="Correlation Matrix")

# Add labels
axis(1, at=1:ncol(cor_matrix), labels=colnames(cor_matrix), las=2, cex.axis=0.8)
axis(2, at=1:nrow(cor_matrix), labels=rownames(cor_matrix), las=2, cex.axis=0.8)

# Add a simple border
box()

dev.off()
cat("Cleaned Correlation Heatmap saved to maciekb/plot_correlation_heatmap.png\n")
