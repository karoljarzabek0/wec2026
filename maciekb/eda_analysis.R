# maciekb/eda_analysis.R
data <- read.csv("for_participants/data/players_quarters_final.csv", stringsAsFactors = FALSE)

# 1. Overall Class Imbalance
total_obs <- nrow(data)
pos_cases <- sum(data$scored_after == 1)
neg_cases <- sum(data$scored_after == 0)
imbalance_pct <- round(100 * pos_cases / total_obs, 2)

cat("--- Overall Class Imbalance ---\n")
cat("Total Observations:", total_obs, "\n")
cat("Positive Cases (scored_after=1):", pos_cases, "(", imbalance_pct, "%)\n")

# 2. Position-wise Goal Probability
pos_stats <- aggregate(scored_after ~ position, data = data, FUN = function(x) {
  c(count = length(x), goals = sum(x), pct = round(100 * mean(x), 2))
})
# Clean up the aggregate output
pos_stats <- data.frame(Position = pos_stats$position, 
                        Total = pos_stats$scored_after[,1], 
                        Goals_Coming_Up = pos_stats$scored_after[,2], 
                        Prob_Pct = pos_stats$scored_after[,3])

cat("\n--- Positional Goal Probability ---\n")
print(pos_stats)

# 3. Scorers vs Non-Scorers: Physical and Technical Metrics
metrics <- c("last15_distance", "last15_sprints", "last15_shots", "last15_shots_on_target")
cat("\n--- Scorers vs Non-Scorers Mean Metrics ---\n")
compare_metrics <- data.frame(Metric = metrics, Scorer_Mean = 0, NonScorer_Mean = 0)

for (i in 1:length(metrics)) {
  m <- metrics[i]
  compare_metrics$Scorer_Mean[i] <- round(mean(data[data$scored_after == 1, m], na.rm=TRUE), 2)
  compare_metrics$NonScorer_Mean[i] <- round(mean(data[data$scored_after == 0, m], na.rm=TRUE), 2)
}
print(compare_metrics)

# 4. Correlation with Target
numeric_cols <- data[, sapply(data, is.numeric)]
correlations <- cor(numeric_cols, data$scored_after, use = "complete.obs")
top_corrs <- sort(correlations[,1], decreasing = TRUE)[2:6] # Top 5 excluding self

cat("\n--- Top Correlations with scored_after ---\n")
print(top_corrs)
