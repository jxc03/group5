# Description: Master entry point for the full project pipeline. Sources
#              every team member's script in the correct order and runs
#              the complete analysis from data loading through to the
#              final cross-model comparison. Run this file to reproduce
#              all results and outputs from scratch
#
# Usage:       Open R/RStudio, set working directory to the project root
#              then run: source("main.R")
#
# Note:        SOURCED_BY_MAIN is set to TRUE before sourcing any modelling
#              script. Each modelling script checks for this flag and skips
#              its standalone execution block, preventing duplicate runs

# Setup
#
# Record the time main.R started so we can report total runtime at the end
start_time <- proc.time()
# Flag that tells every sourced script to skip its standalone execution block
SOURCED_BY_MAIN <- TRUE
# Load shared configuration (paths, seed, CV settings, class labels)
source("config.R")

message(" COM747 CW2 - Heart Disease Prediction Pipeline")
message(" Starting at: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S \n"))

# Helper - Safe Source
# 
# Wraps source() with a file existence check. If a teammate's script is not
# yet available, the pipeline logs a warning and continues rather than
# stopping entirely
safe_source <- function(filepath) {
  if (file.exists(filepath)) {
    message("Sourcing: ", filepath, " ")
    source(filepath)
    message("") # blank line for readability between sections
  } else {
    message("Skipped (file not found): ", filepath, " \n")
  }
}

# Preprocessing [Salah]
# 
# preprocessingScript.R cleans the raw BRFSS CSV, applies outlier capping,
# z-score standardisation, one-hot encoding, Boruta feature selection and
# produces the train/val/test CSV splits including ROSE and SMOTE versions
#
# The output CSVs already exist in data/ so this step only needs to
# be re-run if the raw dataset or preprocessing decisions change
safe_source("scripts/1_preprocessingScript.R")

# Exploratory Data Analysis [Aleemna]
#
# 2_eda.R produces descriptive statistics, class balance summaries and
# all exploratory visualisations. These are saved to outputs/ and used
# in the Results section of the IEEE paper
safe_source("scripts/2_eda.R")

# Logistic Regression [Jonnie]
#
# Trains three LR variants (original, ROSE, SMOTE), selects the best using
# validation AUC, evaluates on the test set and saves plots + model objects.
safe_source("scripts/3_logistic_regression.R")

# Decision Tree [Yasar]
#
# Trains and evaluates the C5.0 Decision Tree model on the test set
# Saves a standardised metrics data frame (dt_results.rds) to outputs/
safe_source("scripts/4_decision_tree.R")

# Random Forest [Aleemna]
#
# Trains and evaluates the Random Forest model, including feature importances
# Saves a standardised metrics data frame (rf_results.rds) to outputs/
safe_source("scripts/5_random_forest.R")

# Cross-Model Evaluation [Jonnie]
# 
# Loads saved results from LR, DT, and RF, combines into one comparison table,
# and produces the ROC overlay plot showing all three models together
# Only runs once all three model results files exist

required_results <- c(
  file.path(CONFIG$output_dir, "lr_results.rds"),
  file.path(CONFIG$output_dir, "dt_results.rds"),
  file.path(CONFIG$output_dir, "rf_results.rds")
)

if (all(file.exists(required_results))) {
  safe_source("scripts/6_evaluations.R")
} else {
  missing <- required_results[!file.exists(required_results)]
  message("--- SKIPPED: 06_evaluation.R ---")
  message("    Waiting on results files from:")
  for (f in missing) message("      - ", f)
  message("")
}

# Finished
elapsed <- proc.time() - start_time

message(" Pipeline complete.")
message(" Total runtime: ", round(elapsed["elapsed"], 1), " seconds")
message(" All outputs saved to: ", CONFIG$output_dir)
message(" Timestamp: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))