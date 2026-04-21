# Description:  Trains and evaluates a Logistic Regression classifier to
#               predict self reported heart disease using the 2020
#               dataset. Loads the pre split CSV files from Salah's
#               preprocessing script, trains three variants (original,
#               ROSE, SMOTE), selects the best using the validation set
#               and evaluates on the held out test set.
# Dependencies: config.R must be sourced before this file.
#               CSV files from preprocessingScript.R must exist at the
#               paths defined in CONFIG.

library(tidyverse) # Data manipulation and ggplot2 plotting
library(caret) # ML interface: train(), confusionMatrix()
library(pROC) # ROC curve and AUC calculation

# Load Data
#
# Reads the five CSV files produced by Salah's preprocessing script and
# ensures HeartDisease is encoded as a factor with "Yes" as the positive class.
#
# Salah's script saves HeartDisease with "No" first. caret treats
# the first factor level as the positive class so without this fix, Sensitivity
# and Specificity would be swapped
#
# @param config - Named list from config.R (file paths, seed, etc.)
# @return Named list: $train, $train_rose, $train_smote, $val, $test
load_preprocessed_data <- function(config) {

  message("[LR] Loading CSV files...")

  train <- read.csv(config$train_path, stringsAsFactors = FALSE)
  train_rose <- read.csv(config$train_rose_path, stringsAsFactors = FALSE)
  train_smote <- read.csv(config$train_smote_path, stringsAsFactors = FALSE)
  val <- read.csv(config$val_path, stringsAsFactors = FALSE)
  test <- read.csv(config$test_path, stringsAsFactors = FALSE)

  # Re-level target so "Yes" = positive class in caret
  relevel_target <- function(df) {
    df$HeartDisease <- factor(df$HeartDisease, levels = c("Yes", "No"))
    return(df)
  }

  train <- relevel_target(train)
  train_rose <- relevel_target(train_rose)
  train_smote <- relevel_target(train_smote)
  val <- relevel_target(val)
  test <- relevel_target(test)

  message("[LR] Class distributions after loading:")
  message(" Original - ", paste(names(table(train$HeartDisease)),
                                   table(train$HeartDisease), sep = ": ", collapse = " | "))
  message(" ROSE - ", paste(names(table(train_rose$HeartDisease)),
                                   table(train_rose$HeartDisease), sep = ": ", collapse = " | "))
  message(" SMOTE - ", paste(names(table(train_smote$HeartDisease)),
                                   table(train_smote$HeartDisease), sep = ": ", collapse = " | "))

  return(list(train = train, train_rose = train_rose,
              train_smote = train_smote, val = val, test = test))
}

# Train and Select Best Variant
#
# Trains a logistic regression model on each of the three training sets
# (original, ROSE balanced, SMOTE balanced) using 10 fold repeated
# cross validation. Each trained model is then evaluated on the shared
# validation set. The variant with the highest validation ROC-AUC is
# returned as the final model.
#
# 9% of cases are positive (heart disease). A model predicting "No" every
# time achieves 91% accuracy but catches zero heart disease cases. AUC
# measures how well the model ranks positives above negatives regardless of
# the threshold
#
# CV is run on the training data, which has been artificially balanced (ROSE/
# SMOTE). Picking the best model from CV alone could favour the balanced
# variants for the wrong reasons. The validation set retains the original
# class distribution making the comparison fair.
#
# @param datasets - Named list from load_preprocessed_data()
# @param config - Named list from config.R
# @return Named list: $best_model, $best_label, $val_aucs, $all_models
train_and_select_lr <- function(datasets, config) {

  # Cross validation settings shared across all three variants
  # repeatedcv: 10-fold CV repeated 3 times gives a more stable AUC estimate
  # classProbs: needed to output probabilities for ROC-AUC scoring
  # twoClassSummary: reports ROC, Sensitivity, Specificity per fold
  cv_control <- trainControl(
    method = "repeatedcv",
    number = config$cv_folds,
    repeats = config$cv_repeats,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    savePredictions = "final"
  )

  # Train each variant using the same CV settings and seed for fair comparison
  training_sets <- list(
    original = datasets$train,
    rose = datasets$train_rose,
    smote = datasets$train_smote
  )

  all_models <- list()
  for (label in names(training_sets)) {
    message("[LR] Training on ", label, " set (", nrow(training_sets[[label]]), " rows)...")
    set.seed(config$seed)  # reset seed before each train() for reproducibility
    all_models[[label]] <- train(
      HeartDisease ~ .,
      data      = training_sets[[label]],
      method    = "glm",
      family    = "binomial",
      trControl = cv_control,
      metric    = "ROC"
    )
    cv <- all_models[[label]]$results
    message("  CV AUC: ", round(cv$ROC, 4),
            " | Sens: ", round(cv$Sens, 4),
            " | Spec: ", round(cv$Spec, 4))
  }

  # Evaluate each model on the validation set and pick the winner
  message("[LR] Selecting best variant using validation set...")
  val_aucs <- sapply(names(all_models), function(label) {
    prob_yes <- predict(all_models[[label]], newdata = datasets$val,
                        type = "prob")[["Yes"]]
    roc_val  <- roc(datasets$val$HeartDisease, prob_yes,
                    levels = c("No", "Yes"), direction = "<", quiet = TRUE)
    val_auc  <- round(as.numeric(auc(roc_val)), 4)
    message("  ", label, " - validation AUC: ", val_auc)
    return(val_auc)
  })

  best_label <- names(which.max(val_aucs))
  message("[LR] Best variant: ", best_label,
          " (AUC = ", max(val_aucs), ")")

  return(list(
    best_model = all_models[[best_label]],
    best_label = best_label,
    val_aucs   = val_aucs,
    all_models = all_models
  ))
}

# Evaluate on Test Set
# 
# Evaluates the selected model on the held-out test set (heart_test.csv)
# This set was never seen during training or model selection
# Produces the full metric suite agreed across the group, in a standardised
# one row data frame that evaluation.R can combine with DT and RF results
#
# @param best_model - Best trained caret model (from train_and_select_lr())
# @param best_label - Label of the winning variant ("original"/"rose"/"smote")
# @param test_df - Held-out test data frame (datasets$test)
# @param config - Named list from config.R
# @return Named list: $metrics, $confusion_matrix, $roc_obj, $predictions
evaluate_lr <- function(best_model, best_label, test_df, config) {

  message("[LR] Evaluating '", best_label, "' model on test set (n = ",
          nrow(test_df), ")...")

  # Predicted class labels (Yes/No) at default 0.5 threshold
  pred_class <- predict(best_model, newdata = test_df, type = "raw")

  # Predicted probabilities for the positive class (needed for ROC curve)
  pred_prob  <- predict(best_model, newdata = test_df, type = "prob")[["Yes"]]

  # Confusion matrix - positive = "Yes" orients TP/FP/FN/TN around heart disease
  cm <- confusionMatrix(
    data = pred_class,
    reference = test_df$HeartDisease,
    positive  = config$positive_class
  )

  # ROC curve object
  roc_obj <- roc(
    response = test_df$HeartDisease,
    predictor = pred_prob,
    levels = c(config$negative_class, config$positive_class),
    direction = "<"
  )

  # One row metrics data frame - rbind(), compatible with DT and RF results
  metrics_df <- data.frame(
    Model = paste0("Logistic Regression (", best_label, ")"),
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Balanced_Accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    Precision = as.numeric(cm$byClass["Pos Pred Value"]),
    F1_Score = as.numeric(cm$byClass["F1"]),
    ROC_AUC = as.numeric(auc(roc_obj)),
    stringsAsFactors = FALSE
  )

  message("[LR] Test set results:")
  print(round(metrics_df[, -1], 4))

  return(list(
    metrics = metrics_df,
    confusion_matrix = cm,
    roc_obj = roc_obj,
    predictions  = data.frame(actual = test_df$HeartDisease,
                              predicted = pred_class,
                              prob_yes = pred_prob)
  ))
}

# Plot Results
# 
# Saves two 300 dpi plots to CONFIG$output_dir for the IEEE paper:
# lr_roc_curve.png - ROC curve with AUC label
# lr_confusion_matrix.png - Confusion matrix heatmap with counts and %
#
# @param lr_results - Named list from evaluate_lr()
# @param config - Named list from config.R
# @return Named list: $roc_plot, $cm_plot (ggplot2 objects)
plot_lr_results <- function(lr_results, config) {

  # ROC Curve
  roc_df <- data.frame(
    fpr = 1 - lr_results$roc_obj$specificities, # x-axis: False Positive Rate
    tpr = lr_results$roc_obj$sensitivities # y-axis: True Positive Rate
  )
  auc_label <- paste0("AUC = ", round(as.numeric(auc(lr_results$roc_obj)), 3))

  roc_plot <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_line(colour = "#1D9E75", linewidth = 1.1) +
    geom_abline(linetype = "dashed", colour = "grey60") +   # random baseline
    annotate("text", x = 0.65, y = 0.15, label = auc_label,
             size = 4.5, fontface = "bold", colour = "#0F6E56") +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    labs(title    = "ROC Curve - Logistic Regression",
         subtitle = paste0(lr_results$metrics$Model, " | BRFSS 2020"),
         x = "1 - Specificity (False Positive Rate)",
         y = "Sensitivity (True Positive Rate)") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  # Confusion Matrix
  cm_df <- as.data.frame(lr_results$confusion_matrix$table)
  n_total <- sum(cm_df$Freq)
  cm_df$label <- paste0(cm_df$Freq, "\n(",
                        round(cm_df$Freq / n_total * 100, 1), "%)")

  cm_plot <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(colour = "white", linewidth = 1.5) +
    geom_text(aes(label = label), size = 4.5, fontface = "bold", colour = "white") +
    scale_fill_gradient(low = "#9FE1CB", high = "#085041", name = "Count") +
    labs(title    = "Confusion Matrix - Logistic Regression",
         subtitle = "Actual vs. predicted | held-out test set",
         x = "Actual Class", y = "Predicted Class") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  # Save 
  roc_path <- file.path(config$output_dir, "lr_roc_curve.png")
  cm_path <- file.path(config$output_dir, "lr_confusion_matrix.png")

  ggsave(roc_path, plot = roc_plot, width = 6, height = 5, dpi = 300)
  ggsave(cm_path, plot = cm_plot,  width = 5, height = 4, dpi = 300)

  message("[LR] Plots saved: ", roc_path, " | ", cm_path)

  return(list(roc_plot = roc_plot, cm_plot = cm_plot))
}

# Standalone execution guard
#
# Runs when this file is sourced directly in RStudio.
# Skipped when main.R sources this file because main.R sets SOURCED_BY_MAIN
# to TRUE beforehand to prevent duplicate training and output conflicts.
#
# Finds config.R whether working directory is the project root or scripts/

if (!exists("SOURCED_BY_MAIN")) {
  
  # Locate config.R, works from both project root and scripts/ subfolder
  config_path <- if (file.exists("config.R")) {
    "config.R"
  } else if (file.exists("../config.R")) {
    "../config.R"
  } else {
    stop("Cannot find config.R. Set your working directory to the project root.")
  }
  source(config_path)
  
  datasets    <- load_preprocessed_data(CONFIG)
  lr_training <- train_and_select_lr(datasets, CONFIG)
  lr_results  <- evaluate_lr(lr_training$best_model,
                             lr_training$best_label,
                             datasets$test, CONFIG)
  lr_plots    <- plot_lr_results(lr_results, CONFIG)
  
  saveRDS(lr_training, file.path(CONFIG$output_dir, "lr_training.rds"))
  saveRDS(lr_results,  file.path(CONFIG$output_dir, "lr_results.rds"))
  
  message("[LR] Done. Outputs saved to: ", CONFIG$output_dir)
}