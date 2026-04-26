# maciekb/eda_mega_analysis.R
library(ggplot2)
library(tidyr)
library(dplyr)

data <- read.csv("for_participants/data/players_quarters_final.csv", stringsAsFactors = FALSE)

# 1. Comprehensive Descriptive Statistics Table
numeric_cols <- data %>% select(starts_with("last15_"), starts_with("cumul_"))
desc_stats <- data.frame(
  Variable = names(numeric_cols),
  Mean = sapply(numeric_cols, mean, na.rm=TRUE),
  SD = sapply(numeric_cols, sd, na.rm=TRUE),
  Min = sapply(numeric_cols, min, na.rm=TRUE),
  Q1 = sapply(numeric_cols, quantile, probs=0.25, na.rm=TRUE),
  Median = sapply(numeric_cols, median, na.rm=TRUE),
  Q3 = sapply(numeric_cols, quantile, probs=0.75, na.rm=TRUE),
  Max = sapply(numeric_cols, max, na.rm=TRUE)
)
write.csv(desc_stats, "maciekb/full_descriptive_stats.csv", row.names=FALSE)

# 2. Positional performance matrix (Means)
pos_matrix <- data %>%
  group_by(position) %>%
  summarise(across(starts_with("last15_"), mean, na.rm=TRUE))
write.csv(pos_matrix, "maciekb/positional_matrix.csv", row.names=FALSE)

# 3. Scoring Probability by Position and Checkpoint (The interaction)
pos_time_prob <- data %>%
  group_by(position, checkpoint) %>%
  summarise(prob = mean(scored_after), .groups = 'drop')
write.csv(pos_time_prob, "maciekb/pos_time_prob.csv", row.names=FALSE)

# 4. PLOTS
# A. Positional Distribution of Distance
p1 <- ggplot(data, aes(x=position, y=last15_distance, fill=position)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title="Distribution of Distance Covered by Position (Last 15 min)", 
       y="Distance (m)", x="Position")
ggsave("maciekb/plot_distance_pos.png", p1, width=8, height=5)

# B. Shots on Target vs Goals Coming Up
p2 <- ggplot(data, aes(x=factor(scored_after), y=last15_shots_on_target, fill=factor(scored_after))) +
  geom_violin() +
  theme_minimal() +
  labs(title="Shots on Target Frequency: Scorers vs Non-Scorers", 
       x="Scored After (Target)", y="Shots on Target (Last 15m)")
ggsave("maciekb/plot_shots_target.png", p2, width=8, height=5)

# C. Temporal Decay of Scoring Probability
p3 <- ggplot(pos_time_prob, aes(x=checkpoint, y=prob, group=position, color=position)) +
  geom_line(size=1) + geom_point() +
  theme_minimal() +
  labs(title="Goal Scoring Probability Decay by Match Time", 
       y="Probability of Scoring After Checkpoint", x="Match Checkpoint")
ggsave("maciekb/plot_temporal_decay.png", p3, width=8, height=5)

cat("EDA Mega-Analysis Complete. Files generated in maciekb/\n")
