# maciekb/check_reset_logic.R
data <- read.csv("for_participants/data/players_quarters_final.csv", stringsAsFactors = FALSE)

# Find players who have scored_after == 1 in the SECOND half
second_half_scorers <- data[grepl("H2", data$checkpoint) & data$scored_after == 1, "player_appearance_id"]
second_half_scorers <- unique(second_half_scorers)

if (length(second_half_scorers) > 0) {
  # Pick a few and see their 1st half flags
  sample_id <- second_half_scorers[1]
  sample_data <- data[data$player_appearance_id == sample_id, ]
  cat("Checking player_appearance_id:", sample_id, "\n")
  print(sample_data[, c("checkpoint", "scored_after")])
  
  # Check overall: How many players have 1 in H2 but 0 in ALL H1?
  h2_only_scorers <- 0
  for (pid in second_half_scorers) {
    p_data <- data[data$player_appearance_id == pid, ]
    h1_flags <- p_data$scored_after[grepl("H1", p_data$checkpoint)]
    if (all(h1_flags == 0, na.rm = TRUE)) {
      h2_only_scorers <- h2_only_scorers + 1
    }
  }
  cat("\nNumber of players who score in H2 but have ALL 0s in H1:", h2_only_scorers, "out of", length(second_half_scorers), "\n")
} else {
  cat("No scorers found in H2.\n")
}
