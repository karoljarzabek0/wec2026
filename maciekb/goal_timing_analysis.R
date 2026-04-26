# maciekb/goal_timing_analysis.R
# Base R version to ensure compatibility

# Load the main dataset
data <- read.csv("for_participants/data/players_quarters_final.csv", stringsAsFactors = FALSE)

# Define the order of checkpoints
checkpoint_order <- c("H1_15", "H1_30", "H1_45", "H2_15", "H2_30", "H2_45", "ET1_15")

# Distribution of 'Scored After' flag across checkpoints
cat("--- Distribution of 'Scored After' by Checkpoint ---\n")
results <- data.frame(checkpoint = checkpoint_order, 
                      total_players = 0, 
                      goals_coming_up = 0)

for (i in 1:length(checkpoint_order)) {
  cp <- checkpoint_order[i]
  subset_data <- data[data$checkpoint == cp, ]
  results$total_players[i] <- nrow(subset_data)
  results$goals_coming_up[i] <- sum(subset_data$scored_after, na.rm = TRUE)
}
results$percentage <- round(100 * results$goals_coming_up / results$total_players, 2)
print(results)

# Inferred Goal Timing
# Logic: Goal happened in interval T if (scored_after == 1 at T-1) AND (scored_after == 0 at T)
cat("\n--- Inferred Goals Scored per 15-minute Interval ---\n")
cat("(Interval ends at the checkpoint listed)\n")

# Sort data by player and checkpoint order
data$checkpoint_idx <- match(data$checkpoint, checkpoint_order)
data <- data[order(data$player_appearance_id, data$checkpoint_idx), ]

inferred_counts <- data.frame(checkpoint = checkpoint_order, goals_scored = 0)

# We start from the second checkpoint to find transitions
unique_players <- unique(data$player_appearance_id)

for (pid in unique_players) {
  p_data <- data[data$player_appearance_id == pid, ]
  if (nrow(p_data) < 2) next
  
  for (j in 2:nrow(p_data)) {
    if (p_data$scored_after[j-1] == 1 && p_data$scored_after[j] == 0) {
      cp_name <- as.character(p_data$checkpoint[j])
      idx <- which(inferred_counts$checkpoint == cp_name)
      inferred_counts$goals_scored[idx] <- inferred_counts$goals_scored[idx] + 1
    }
  }
}

print(inferred_counts[inferred_counts$goals_scored > 0, ])

cat("\nNote: Goals scored in the first 15 mins (01-15 H1) cannot be detected this way\n")
cat("because we don't have a 'pre-match' checkpoint. However, the data shows that\n")
cat("the number of players with goals 'coming up' is highest at the start.\n")
