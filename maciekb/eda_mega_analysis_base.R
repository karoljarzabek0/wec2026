# maciekb/eda_mega_analysis_base.R

data <- read.csv("for_participants/data/players_quarters_final.csv", stringsAsFactors = FALSE)

# 1. Comprehensive Descriptive Statistics Table
numeric_col_names <- grep("^last15_|^cumul_", names(data), value = TRUE)
numeric_cols <- data[, numeric_col_names]

desc_stats <- data.frame(
  Variable = numeric_col_names,
  Mean = sapply(numeric_cols, mean, na.rm=TRUE),
  SD = sapply(numeric_cols, sd, na.rm=TRUE),
  Min = sapply(numeric_cols, min, na.rm=TRUE),
  Q1 = sapply(numeric_cols, function(x) quantile(x, probs=0.25, na.rm=TRUE)),
  Median = sapply(numeric_cols, median, na.rm=TRUE),
  Q3 = sapply(numeric_cols, function(x) quantile(x, probs=0.75, na.rm=TRUE)),
  Max = sapply(numeric_cols, max, na.rm=TRUE)
)
write.csv(desc_stats, "maciekb/full_descriptive_stats.csv", row.names=FALSE)

# 2. Positional performance matrix (Means)
last15_cols <- grep("^last15_", names(data), value = TRUE)
pos_matrix <- aggregate(data[, last15_cols], list(position = data$position), mean, na.rm=TRUE)
write.csv(pos_matrix, "maciekb/positional_matrix.csv", row.names=FALSE)

# 3. Scoring Probability by Position and Checkpoint
pos_time_prob <- aggregate(scored_after ~ position + checkpoint, data = data, FUN = mean, na.rm=TRUE)
print("pos_time_prob head:")
print(head(pos_time_prob))
names(pos_time_prob)[3] <- "prob"
write.csv(pos_time_prob, "maciekb/pos_time_prob.csv", row.names=FALSE)

# 4. PLOTS
# A. Positional Distribution of Distance
print("Generating Plot A")
png("maciekb/plot_distance_pos.png", width=800, height=500)
boxplot(last15_distance ~ position, data = data,
        main="Distribution of Distance Covered by Position (Last 15 min)",
        ylab="Distance (m)", xlab="Position", col=rainbow(length(unique(data$position))))
dev.off()

# B. Shots on Target vs Goals Coming Up
print("Generating Plot B")
png("maciekb/plot_shots_target.png", width=800, height=500)
boxplot(last15_shots_on_target ~ scored_after, data = data,
        main="Shots on Target Frequency: Scorers vs Non-Scorers",
        xlab="Scored After (Target)", ylab="Shots on Target (Last 15m)",
        col=c("red", "green"))
dev.off()

# C. Temporal Decay of Scoring Probability
print("Generating Plot C")
positions <- unique(pos_time_prob$position)

checkpoint_map <- data.frame(
  checkpoint = c("H1_15", "H1_30", "H1_45", "H2_15", "H2_30", "H2_45", "ET1_15", "ET2_15"),
  time = c(15, 30, 45, 60, 75, 90, 105, 120)
)
pos_time_prob <- merge(pos_time_prob, checkpoint_map, by = "checkpoint", all.x = TRUE)
pos_time_prob <- pos_time_prob[!is.na(pos_time_prob$time), ]

checkpoints_numeric <- sort(unique(pos_time_prob$time))
print("Numeric Checkpoints:")
print(checkpoints_numeric)

png("maciekb/plot_temporal_decay.png", width=800, height=500)
plot(NULL, xlim=range(checkpoints_numeric), ylim=range(pos_time_prob$prob, na.rm=TRUE),
     main="Goal Scoring Probability Decay by Match Time",
     ylab="Probability of Scoring After Checkpoint", xlab="Match Time (min)",
     xaxt="n")
axis(1, at=checkpoint_map$time, labels=checkpoint_map$checkpoint)

colors <- rainbow(length(positions))
for (i in seq_along(positions)) {
  sub_data <- pos_time_prob[pos_time_prob$position == positions[i], ]
  sub_data <- sub_data[order(sub_data$time), ]
  lines(sub_data$time, sub_data$prob, col=colors[i], lwd=2, type="b", pch=19)
}
legend("topright", legend=positions, col=colors, lwd=2, pch=19)
dev.off()

cat("EDA Mega-Analysis (Base R) Complete. Files generated in maciekb/\n")
