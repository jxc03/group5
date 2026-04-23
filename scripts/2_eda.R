# 2_eda.R
# Description: Performs exploratory data analysis on the shared preprocessed
#              training data used by the group models and saves plots/statistics.
# Dependencies: config.R must be sourced before this file.

library(tidyverse)

# Load Data
#
# Reads the shared train/test CSV files from the preprocessing pipeline.
#
# @param config Named list from config.R
# @return Named list: $train, $test
load_eda_data <- function(config) {
  
  message("[EDA] Loading CSV files...")
  
  train <- read.csv(config$train_path, stringsAsFactors = FALSE)
  test  <- read.csv(config$test_path, stringsAsFactors = FALSE)
  
  train$HeartDisease <- factor(train$HeartDisease, levels = c("Yes", "No"))
  test$HeartDisease  <- factor(test$HeartDisease, levels = c("Yes", "No"))
  
  return(list(train = train, test = test))
}

# Run EDA
#
# Produces descriptive statistics and association tests on the training set.
#
# @param train_df Training data frame
# @param config Named list from config.R
# @return Named list with summaries and tests
run_eda <- function(train_df, config) {
  
  message("[EDA] Running descriptive statistics and tests...")
  
  bmi_summary <- train_df %>%
    group_by(HeartDisease) %>%
    summarise(
      mean_bmi = mean(BMI, na.rm = TRUE),
      sd_bmi = sd(BMI, na.rm = TRUE),
      median_bmi = median(BMI, na.rm = TRUE),
      .groups = "drop"
    )
  
  smoking_table <- prop.table(table(train_df$Smoking, train_df$HeartDisease), margin = 1)
  sex_table <- prop.table(table(train_df$Sex, train_df$HeartDisease), margin = 1)
  genhealth_table <- prop.table(table(train_df$GenHealth, train_df$HeartDisease), margin = 1)
  
  bmi_ttest <- t.test(BMI ~ HeartDisease, data = train_df)
  smoking_chisq <- chisq.test(table(train_df$Smoking, train_df$HeartDisease))
  sex_chisq <- chisq.test(table(train_df$Sex, train_df$HeartDisease))
  genhealth_chisq <- chisq.test(table(train_df$GenHealth, train_df$HeartDisease))
  
  eda_results <- list(
    bmi_summary = bmi_summary,
    smoking_table = smoking_table,
    sex_table = sex_table,
    genhealth_table = genhealth_table,
    bmi_ttest = bmi_ttest,
    smoking_chisq = smoking_chisq,
    sex_chisq = sex_chisq,
    genhealth_chisq = genhealth_chisq
  )
  
  saveRDS(eda_results, file.path(config$output_dir, "eda_results.rds"))
  
  return(eda_results)
}

# Plot EDA results
#
# Saves core EDA figures for the report.
#
# @param train_df Training data frame
# @param config Named list from config.R
# @return Named list of ggplot objects
plot_eda_results <- function(train_df, config) {
  
  p1 <- ggplot(train_df, aes(x = HeartDisease)) +
    geom_bar(fill = "steelblue") +
    labs(
      title = "Distribution of Heart Disease",
      x = "Heart Disease",
      y = "Count"
    ) +
    theme_minimal(base_size = 12)
  
  p2 <- ggplot(train_df, aes(x = HeartDisease, y = BMI)) +
    geom_boxplot(fill = "lightblue") +
    labs(
      title = "BMI by Heart Disease Status",
      x = "Heart Disease",
      y = "BMI"
    ) +
    theme_minimal(base_size = 12)
  
  p3 <- ggplot(train_df, aes(x = AgeCategory, fill = HeartDisease)) +
    geom_bar(position = "fill") +
    labs(
      title = "Heart Disease by Age Category",
      x = "Age Category",
      y = "Proportion"
    ) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  p4 <- ggplot(train_df, aes(x = Smoking, fill = HeartDisease)) +
    geom_bar(position = "fill") +
    labs(
      title = "Heart Disease by Smoking Status",
      x = "Smoking",
      y = "Proportion"
    ) +
    theme_minimal(base_size = 12)
  
  p5 <- ggplot(train_df, aes(x = GenHealth, fill = HeartDisease)) +
    geom_bar(position = "fill") +
    labs(
      title = "Heart Disease by General Health",
      x = "General Health",
      y = "Proportion"
    ) +
    theme_minimal(base_size = 12)
  
  p6 <- ggplot(train_df, aes(x = Sex, fill = HeartDisease)) +
    geom_bar(position = "fill") +
    labs(
      title = "Heart Disease by Sex",
      x = "Sex",
      y = "Proportion"
    ) +
    theme_minimal(base_size = 12)
  
  ggsave(file.path(config$output_dir, "heart_disease_distribution.png"), plot = p1, width = 6, height = 5, dpi = 300)
  ggsave(file.path(config$output_dir, "bmi_vs_heart_disease.png"), plot = p2, width = 6, height = 5, dpi = 300)
  ggsave(file.path(config$output_dir, "age_vs_heart_disease.png"), plot = p3, width = 7, height = 5, dpi = 300)
  ggsave(file.path(config$output_dir, "smoking_vs_heart_disease.png"), plot = p4, width = 6, height = 5, dpi = 300)
  ggsave(file.path(config$output_dir, "genhealth_vs_heart_disease.png"), plot = p5, width = 6, height = 5, dpi = 300)
  ggsave(file.path(config$output_dir, "sex_vs_heart_disease.png"), plot = p6, width = 6, height = 5, dpi = 300)
  
  message("[EDA] Plots saved to: ", config$output_dir)
  
  return(list(
    target_plot = p1,
    bmi_plot = p2,
    age_plot = p3,
    smoking_plot = p4,
    genhealth_plot = p5,
    sex_plot = p6
  ))
}

# Standalone execution guard
if (!exists("SOURCED_BY_MAIN")) {
  
  config_path <- if (file.exists("config.R")) {
    "config.R"
  } else if (file.exists("../config.R")) {
    "../config.R"
  } else {
    stop("Cannot find config.R. Set your working directory to the project root.")
  }
  source(config_path)
  
  eda_data <- load_eda_data(CONFIG)
  eda_results <- run_eda(eda_data$train, CONFIG)
  eda_plots <- plot_eda_results(eda_data$train, CONFIG)
  
  message("[EDA] Done. Outputs saved to: ", CONFIG$output_dir)
}