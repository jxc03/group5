# Description:  Trains and evaluates a C5.0 Decision Tree classifier to
#               predict self reported heart disease using the preprocessed
#               BRFSS 2020 dataset. Saves results in the same structure used
#               by the other modelling scripts so evaluation.R can combine them.
# Dependencies: config.R must be sourced before standalone execution.

library(C50)
library(caret)
library(pROC)

# Load Data
#
# Reads the original training and held-out test sets produced by the
# preprocessing script and ensures HeartDisease uses the agreed factor order
# with "Yes" as the positive class.
#
# @param config - Named list from config.R
# @return Named list: $train, $test
load_dt_data <- function(config) {
  
  message("[DT] Loading CSV files...")
  
  train <- read.csv(config$train_path, stringsAsFactors = FALSE)
  test  <- read.csv(config$test_path, stringsAsFactors = FALSE)
  
  relevel_target <- function(df) {
    df$HeartDisease <- factor(df$HeartDisease, levels = c("Yes", "No"))
    return(df)
  }
  
  train <- relevel_target(train)
  test  <- relevel_target(test)
  
  return(list(train = train, test = test))
}

# Train C5.0 Model
#
# @param train_data - Training data frame
# @return Trained C5.0 model object
train_c50_model <- function(train_data) {
  
  message("[DT] Training C5.0 model on ", nrow(train_data), " rows...")
  
  model <- C5.0(HeartDisease ~ ., data = train_data, trials = 10)
  
  return(model)
}

# Evaluate on Test Set
#
# Produces the same result structure used by evaluate_lr() so the shared
# evaluation script can consume this file without any special casing.
#
# @param model - Trained C5.0 model
# @param test_data - Held-out test data frame
# @param config - Named list from config.R
# @return Named list: $metrics, $confusion_matrix, $roc_obj, $predictions, $model
#
evaluate_dt <- function(model, test_data, config) {
  
  message("[DT] Evaluating model on test set (n = ", nrow(test_data), ")...")
  
  pred_class <- predict(model, newdata = test_data, type = "class")
  pred_prob  <- predict(model, newdata = test_data, type = "prob")[, "Yes"]
  
  cm <- confusionMatrix(
    data = factor(pred_class, levels = c(config$positive_class, config$negative_class)),
    reference = test_data$HeartDisease,
    positive = config$positive_class
  )
  
  roc_obj <- roc(
    response = test_data$HeartDisease,
    predictor = pred_prob,
    levels = c(config$negative_class, config$positive_class),
    direction = "<"
  )
  
  metrics_df <- data.frame(
    Model = "Decision Tree (C5.0)",
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Balanced_Accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    Precision = as.numeric(cm$byClass["Pos Pred Value"]),
    F1_Score = as.numeric(cm$byClass["F1"]),
    ROC_AUC = as.numeric(auc(roc_obj)),
    stringsAsFactors = FALSE
  )
  
  message("[DT] Test set results:")
  print(round(metrics_df[, -1], 4))
  
  return(list(
    model = model,
    metrics = metrics_df,
    confusion_matrix = cm,
    roc_obj = roc_obj,
    predictions = data.frame(
      actual = test_data$HeartDisease,
      predicted = pred_class,
      prob_yes = pred_prob
    )
  ))
}

# STANDALONE EXECUTION GUARD
#
# Runs when this file is sourced directly in RStudio. Skipped when main.R
# sources this file because main.R sets SOURCED_BY_MAIN to TRUE first.

if (!exists("SOURCED_BY_MAIN")) {
  
  source("config.R")
  
  datasets <- load_dt_data(CONFIG)
  dt_model <- train_c50_model(datasets$train)
  dt_results <- evaluate_dt(dt_model, datasets$test, CONFIG)
  
  saveRDS(dt_results, file.path(CONFIG$output_dir, "dt_results.rds"))
  
  message("[DT] Done. Outputs saved to: ", CONFIG$output_dir)
}