# Description: Central configuration file. All shared settings (file paths,
#              random seed, CV parameters, class labels) are defined here
#              Every modelling script begins with: source("config.R")
#              Changing a value here updates the entire project automatically

CONFIG <- list(

  # File paths - produced bypreprocessingScript.R
  train_path = "data/heart_train.csv",
  train_rose_path  = "data/heart_train_rose.csv",
  train_smote_path = "data/heart_train_smote.csv",
  val_path = "data/heart_val.csv",
  test_path = "data/heart_test.csv",

  # All generated plots and saved model objects go here
  output_dir = "outputs/",

  # Reproducibility
  # 
  # Salah's preprocessing used seed 42 - we keep this consistent across all
  # scripts so that CV fold assignment and any other random operations are
  # identical every time the project is run
  seed = 42,

  # Cross-validation
  # 
  # 10 fold CV repeated 3 times, standard for medium/large tabular datasets
  cv_folds = 10,
  cv_repeats = 3,

  # Class labels
  # 
  # "Yes" is the positive class (heart disease present)
  # This must match the factor relevelling done in each modelling script
  positive_class   = "Yes",
  negative_class   = "No"
)

# Create the output directory if it does not already exist
if (!dir.exists(CONFIG$output_dir)) {
  dir.create(CONFIG$output_dir, recursive = TRUE)
  message("[Config] Created output directory: ", CONFIG$output_dir)
}

message("[Config] Loaded successfully. Seed = ", CONFIG$seed)
