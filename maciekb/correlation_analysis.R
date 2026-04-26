# maciekb/correlation_analysis.R
data <- read.csv("for_participants/data/players_quarters_final.csv", stringsAsFactors = FALSE)

# Select only numeric performance columns
numeric_cols <- data[, sapply(data, is.numeric)]
# Remove ID columns that aren't performance related
performance_cols <- numeric_cols[, !names(numeric_cols) %in% c("player_appearance_id", "player_id", "fixture_id", "jersey_number")]

# Calculate correlation matrix
cor_matrix <- cor(performance_cols, use = "complete.obs")

# Round for readability and save
cor_matrix_rounded <- round(cor_matrix, 3)
write.csv(cor_matrix_rounded, "maciekb/full_correlation_matrix.csv")

# Identify the strongest correlations (excluding 1.0 diagonal)
cor_tri <- cor_matrix
cor_tri[lower.tri(cor_tri, diag = TRUE)] <- NA
cor_list <- as.data.frame(as.table(cor_tri))
cor_list <- na.omit(cor_list)
cor_list <- cor_list[order(abs(cor_list$Freq), decreasing = TRUE), ]

cat("--- Top 10 Strongest Inter-Variable Correlations ---\n")
print(head(cor_list, 10))

# Create a basic heatmap using Base R
png("maciekb/plot_correlation_heatmap.png", width = 800, height = 800)
heatmap(cor_matrix, 
        main = "Correlation Heatmap: Performance Metrics",
        col = cm.colors(256), 
        scale = "none", 
        margins = c(10, 10))
dev.off()

cat("\nFull correlation matrix saved to maciekb/full_correlation_matrix.csv\n")
cat("Heatmap saved to maciekb/plot_correlation_heatmap.png\n")
