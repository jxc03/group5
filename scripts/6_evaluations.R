# Description:  Cross model evaluation script. Loads the saved results from
#               each model (Logistic Regression, Decision Tree, Random Forest),
#               combines metrics into a single comparison table and produces
#               an ROC overlay plot showing all three models together
#               Outputs are saved to CONFIG$output_dir for use in the paper
# Dependencies: Expects result .rds files saved by each modelling script

library(tidyverse) # ggplot2 for plotting, dplyr for table formatting
library(pROC)# roc() and auc() for ROC overlay

# Load Model Results
# 
# Loads the saved results .rds files produced by each modelling script
# Uses the same safe loading pattern as main.R, if a model's results file
# does not exist yet, it is skipped with a warning rather than crashing
#
# @param config - Named list from config.R
# @return - Named list of result objects, one entry per available model
load_model_results <- function(config) {

  result_files <- list(
    "Logistic Regression" = file.path(config$output_dir, "lr_results.rds"),
    "Decision Tree" = file.path(config$output_dir, "dt_results.rds"),
    "Random Forest" = file.path(config$output_dir, "rf_results.rds")
  )

  results <- list()

  for (model_name in names(result_files)) {
    path <- result_files[[model_name]]
    if (file.exists(path)) {
      results[[model_name]] <- readRDS(path)
      message("[Eval] Loaded: ", model_name)
    } else {
      message("[Eval] Skipped (not found): ", model_name, " - ", path)
    }
  }

  if (length(results) == 0) {
    stop("[Eval] No model results found in ", config$output_dir,
         ". Run modelling scripts first.")
  }

  message("[Eval] ", length(results), " model(s) loaded for comparison.")
  return(results)
}

# Build Comparison Table
# 
# Combines the one row metrics data frame from each model into a single
# comparison table. Rounds all numeric values to 4 decimal places for
# consistency, prints a formatted summary to the console and saves the
# table as a CSV to CONFIG$output_dir.
#
# The table format is rbind compatible because each modelling script was
# designed to return metrics in the same standardised column structure:
# Model | Accuracy | Balanced_Accuracy | Sensitivity | Specificity |
# Precision | F1_Score | ROC_AUC
#
# @param results - Named list from load_model_results()
# @param config - Named list from config.R
# @return - Data frame of combined metrics (one row per model)
build_comparison_table <- function(results, config) {

  message("[Eval] Building comparison table...")

  # Extract the $metrics data frame from each result object and stack them
  metrics_list <- lapply(results, function(r) r$metrics)
  comparison   <- do.call(rbind, metrics_list)
  rownames(comparison) <- NULL # clean row names after rbind

  # Round all numeric columns to 4 decimal places for readability
  numeric_cols <- sapply(comparison, is.numeric)
  comparison[numeric_cols] <- round(comparison[numeric_cols], 4)

  # Print a clean summary to the console
  message("\n[Eval] Model Comparison (held-out test set) ")
  print(comparison, row.names = FALSE)

  # Highlight the best value per metric so it is easy to spot
  message("\n[Eval] Best model per metric ")
  for (col in names(comparison)[numeric_cols]) {
    best_idx   <- which.max(comparison[[col]])
    best_model <- comparison$Model[best_idx]
    best_val   <- comparison[[col]][best_idx]
    message("  ", col, ": ", best_model, " (", best_val, ")")
  }

  # Save as CSV for easy reference and potential table inclusion in the paper
  out_path <- file.path(config$output_dir, "model_comparison.csv")
  write.csv(comparison, out_path, row.names = FALSE)
  message("\n[Eval] Comparison table saved: ", out_path)

  return(comparison)
}

# ROC Overlay Plot
# 
# Produces a single plot with the ROC curve of every available model drawn
# together, each in a distinct colour with the AUC shown in the legend
# This is the key visualisation for the Results section of the IEEE paper
# as it allows direct visual comparison of all three classifiers
#
# Saved as evaluation_roc_overlay.png (300 dpi) to CONFIG$output_dir.
#
# @param results - Named list from load_model_results()
# @param config - Named list from config.R
# @return - ggplot2 object (also saved to disk)
plot_roc_overlay <- function(results, config) {

  message("[Eval] Building ROC overlay plot...")

  # Colour palette - one distinct colour per model, consistent with the paper
  model_colours <- c(
    "Logistic Regression" = "#1D9E75", # teal
    "Decision Tree" = "#7F77DD", # purple
    "Random Forest" = "#EF9F27" # amber
  )

  # Build a single data frame with FPR, TPR, model name and AUC label
  # so ggplot2 can handle all curves in one geom_line() call
  roc_df <- lapply(names(results), function(model_name) {
    roc_obj <- results[[model_name]]$roc_obj
    auc_val <- round(as.numeric(auc(roc_obj)), 3)
    # Build the legend label to include the AUC value
    leg_label <- paste0(model_name, "  (AUC = ", auc_val, ")")
    data.frame(
      fpr = 1 - roc_obj$specificities,
      tpr = roc_obj$sensitivities,
      model = leg_label
    )
  })
  roc_df <- do.call(rbind, roc_df)

  # Map the longer legend labels back to the palette colours
  # Match on the model name prefix so colour assignment still works
  # even though the legend label now includes the AUC string
  legend_colours <- setNames(
    model_colours[match(
      sub("  \\(AUC.*", "", unique(roc_df$model)), # strip AUC from label
      names(model_colours)
    )],
    unique(roc_df$model)
  )

  roc_plot <- ggplot(roc_df, aes(x = fpr, y = tpr,
                                 colour = model, group = model)) +
    geom_line(linewidth = 1.1) +
    # Dashed diagonal = random classifier baseline (AUC = 0.50)
    geom_abline(linetype = "dashed", colour = "grey60", linewidth = 0.7) +
    scale_colour_manual(values = legend_colours, name = NULL) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    labs(
      title = "ROC Curve Comparison - All Models",
      subtitle = "Predicting self-reported heart disease | BRFSS 2020 | held-out test set",
      x = "1 - Specificity (False Positive Rate)",
      y = "Sensitivity (True Positive Rate)"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = c(0.72, 0.18),  # inside plot, bottom right
      legend.background = element_rect(fill = "white", colour = "grey90",
                                       linewidth = 0.4),
      legend.margin = margin(6, 8, 6, 8)
    )

  out_path <- file.path(config$output_dir, "evaluation_roc_overlay.png")
  ggsave(out_path, plot = roc_plot, width = 6, height = 5, dpi = 300)
  message("[Eval] ROC overlay saved: ", out_path)

  return(roc_plot)
}

# Bar Chart of Key Metrics
# 
# Produces a grouped bar chart comparing Balanced Accuracy, Sensitivity,
# Specificity, F1 and ROC AUC across all models side by side
# This gives a clearer view of trade-offs than a table alone
# Saved as evaluation_metrics_barchart.png to CONFIG$output_dir
#
# @param comparison - Data frame from build_comparison_table()
# @param config- Named list from config.R
# @return - ggplot2 object (also saved to disk)
plot_metrics_barchart <- function(comparison, config) {

  message("[Eval] Building metrics bar chart...")

  # Select the metrics most relevant to imbalanced classification
  # (Accuracy alone is misleading - see logistic regression script comments)
  key_metrics <- c("Balanced_Accuracy", "Sensitivity",
                   "Specificity", "F1_Score", "ROC_AUC")

  # Pivot to long format so ggplot2 can group by metric
  long_df <- comparison %>%
    select(Model, all_of(key_metrics)) %>%
    pivot_longer(cols = all_of(key_metrics),
                 names_to = "Metric",
                 values_to = "Value") %>%
    # Clean up metric labels for the plot axis
    mutate(Metric = recode(Metric,
      "Balanced_Accuracy" = "Balanced Acc.",
      "Sensitivity" = "Sensitivity",
      "Specificity" = "Specificity",
      "F1_Score" = "F1 Score",
      "ROC_AUC" = "ROC AUC"
    ))

  bar_plot <- ggplot(long_df, aes(x = Metric, y = Value, fill = Model)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    geom_text(aes(label = round(Value, 3)),
              position = position_dodge(width = 0.75),
              vjust = -0.4, size = 3, fontface = "bold") +
    scale_fill_manual(
      values = c(
        # Partial match on model name prefix in case label includes variant
        setNames(
          c("#1D9E75", "#7F77DD", "#EF9F27"),
          unique(grep("Logistic|Decision|Random",
                      comparison$Model, value = TRUE))
        )
      ),
      name = NULL
    ) +
    scale_y_continuous(limits = c(0, 1.08), expand = c(0, 0)) +
    labs(
      title = "Key Metrics Comparison - All Models",
      subtitle = "Held-out test set | BRFSS 2020",
      x = NULL,
      y = "Score"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "bottom",
      panel.grid.major.x = element_blank()
    )

  out_path <- file.path(config$output_dir, "evaluation_metrics_barchart.png")
  ggsave(out_path, plot = bar_plot, width = 8, height = 5, dpi = 300)
  message("[Eval] Metrics bar chart saved: ", out_path)

  return(bar_plot)
}

# Standalone execution guard
if (!exists("SOURCED_BY_MAIN")) {

  source("config.R")

  results <- load_model_results(CONFIG)
  comparison <- build_comparison_table(results, CONFIG)
  roc_plot <- plot_roc_overlay(results, CONFIG)
  bar_plot <- plot_metrics_barchart(comparison, CONFIG)

  message("\n[Eval] Complete. All evaluation outputs saved to: ",
          CONFIG$output_dir)
}