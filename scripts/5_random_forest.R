# 5_random_forest.R
# Description: Trains and evaluates a Random Forest classifier to predict
#              self-reported heart disease using the shared preprocessed
#              train/test CSV files from the group pipeline.
# Dependencies: config.R must be sourced before this file.
#               CSV files from preprocessingScript.R must exist at paths in CONFIG.

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)

# Load Data
#
# Reads the preprocessed train and test CSV files and ensures HeartDisease
# is encoded with "Yes" as the positive class.
#
# @param config Named list from config.R
# @return Named list: $train, $test
load_rf_data <- function(config) {
  
  message("[RF] Loading CSV files...")
  
  train <- read.csv(config$train_path, stringsAsFactors = FALSE)
  test  <- read.csv(config$test_path, stringsAsFactors = FALSE)
  
  relevel_target <- function(df) {
    df$HeartDisease <- factor(df$HeartDisease, levels = c("Yes", "No"))
    return(df)
  }
  
  train <- relevel_target(train)
  test  <- relevel_target(test)
  
  message("[RF] Class distributions after loading:")
  message(" Train - ", paste(names(table(train$HeartDisease)),
                             table(train$HeartDisease), sep = ": ", collapse = " | "))
  message(" Test - ", paste(names(table(test$HeartDisease)),
                            table(test$HeartDisease), sep = ": ", collapse = " | "))
  
  return(list(train = train, test = test))
}

# Train Random Forest
#
# Trains a Random Forest classifier on the shared training set.
#
# @param train_df Training data frame
# @param config Named list from config.R
# @return Trained randomForest model
train_random_forest <- function(train_df, config) {
  
  message("[RF] Training Random Forest on training set (n = ", nrow(train_df), ")...")
  
  # Convert character columns to factors
  train_df[] <- lapply(train_df, function(x) {
    if (is.character(x)) as.factor(x) else x
  })
  
  set.seed(config$seed)
  
  rf_model <- randomForest(
    HeartDisease ~ .,
    data = train_df,
    ntree = 200,
    importance = TRUE
  )
  
  message("[RF] Training complete.")
  return(rf_model)
}

# Evaluate on Test Set
#
# Evaluates the Random Forest model on the held-out test set.
#
# @param rf_model Trained randomForest model
# @param test_df Held-out test data frame
# @param config Named list from config.R
# @return Named list: $metrics, $confusion_matrix, $roc_obj, $predictions, $feature_importance
evaluate_rf <- function(rf_model, test_df, config) {
  
  message("[RF] Evaluating model on test set (n = ", nrow(test_df), ")...")
  
  # Convert character columns to factors
  test_df[] <- lapply(test_df, function(x) {
    if (is.character(x)) as.factor(x) else x
  })
  
  test_df$HeartDisease <- factor(test_df$HeartDisease,
                                 levels = c(config$positive_class, config$negative_class))
  
  pred_class <- predict(rf_model, newdata = test_df, type = "response")
  pred_prob  <- predict(rf_model, newdata = test_df, type = "prob")[["Yes"]]
  
  cm <- confusionMatrix(
    data = pred_class,
    reference = test_df$HeartDisease,
    positive = config$positive_class
  )
  
  roc_obj <- roc(
    response = test_df$HeartDisease,
    predictor = pred_prob,
    levels = c(config$negative_class, config$positive_class),
    direction = "<"
  )
  
  # Feature importance table
  rf_importance <- as.data.frame(importance(rf_model))
  rf_importance$Feature <- rownames(rf_importance)
  rf_importance <- rf_importance[order(-rf_importance$MeanDecreaseGini), ]
  
  metrics_df <- data.frame(
    Model = "Random Forest",
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Balanced_Accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    Precision = as.numeric(cm$byClass["Pos Pred Value"]),
    F1_Score = as.numeric(cm$byClass["F1"]),
    ROC_AUC = as.numeric(auc(roc_obj)),
    stringsAsFactors = FALSE
  )
  
  message("[RF] Test set results:")
  print(round(metrics_df[, -1], 4))
  
  return(list(
    metrics = metrics_df,
    confusion_matrix = cm,
    roc_obj = roc_obj,
    predictions = data.frame(
      actual = test_df$HeartDisease,
      predicted = pred_class,
      prob_yes = pred_prob
    ),
    feature_importance = rf_importance
  ))
}

# Plot Results
#
# Saves two plots to CONFIG$output_dir:
# rf_roc_curve.png
# rf_feature_importance.png
#
# @param rf_results Named list from evaluate_rf()
# @param config Named list from config.R
# @return Named list: $roc_plot, $importance_plot
plot_rf_results <- function(rf_results, config) {
  
  # ROC plot
  roc_df <- data.frame(
    fpr = 1 - rf_results$roc_obj$specificities,
    tpr = rf_results$roc_obj$sensitivities
  )
  
  auc_label <- paste0("AUC = ", round(as.numeric(auc(rf_results$roc_obj)), 3))
  
  roc_plot <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_line(colour = "#C44E52", linewidth = 1.1) +
    geom_abline(linetype = "dashed", colour = "grey60") +
    annotate("text", x = 0.65, y = 0.15, label = auc_label,
             size = 4.5, fontface = "bold", colour = "#8B1E22") +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    labs(
      title = "ROC Curve - Random Forest",
      subtitle = "BRFSS 2020",
      x = "1 - Specificity (False Positive Rate)",
      y = "Sensitivity (True Positive Rate)"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))
  
  # Feature importance plot
  top_features <- rf_results$feature_importance %>%
    dplyr::select(Feature, MeanDecreaseGini) %>%
    slice_head(n = 10) %>%
    mutate(Feature = reorder(Feature, MeanDecreaseGini))
  
  importance_plot <- ggplot(top_features, aes(x = Feature, y = MeanDecreaseGini)) +
    geom_col(fill = "#4C78A8") +
    coord_flip() +
    labs(
      title = "Feature Importance - Random Forest",
      subtitle = "Top 10 predictors by Mean Decrease Gini",
      x = "Feature",
      y = "Mean Decrease Gini"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))
  
  roc_path <- file.path(config$output_dir, "rf_roc_curve.png")
  importance_path <- file.path(config$output_dir, "rf_feature_importance.png")
  
  ggsave(roc_path, plot = roc_plot, width = 6, height = 5, dpi = 300)
  ggsave(importance_path, plot = importance_plot, width = 6, height = 5, dpi = 300)
  
  write.csv(
    rf_results$metrics,
    file.path(config$output_dir, "random_forest_metrics.csv"),
    row.names = FALSE
  )
  
  write.csv(
    rf_results$feature_importance,
    file.path(config$output_dir, "random_forest_feature_importance.csv"),
    row.names = FALSE
  )
  
  message("[RF] Plots saved: ", roc_path, " | ", importance_path)
  
  return(list(
    roc_plot = roc_plot,
    importance_plot = importance_plot
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
  
  rf_data <- load_rf_data(CONFIG)
  rf_model <- train_random_forest(rf_data$train, CONFIG)
  rf_results <- evaluate_rf(rf_model, rf_data$test, CONFIG)
  rf_plots <- plot_rf_results(rf_results, CONFIG)
  
  saveRDS(rf_model, file.path(CONFIG$output_dir, "rf_model.rds"))
  saveRDS(rf_results, file.path(CONFIG$output_dir, "rf_results.rds"))
  
  message("[RF] Done. Outputs saved to: ", CONFIG$output_dir)
}