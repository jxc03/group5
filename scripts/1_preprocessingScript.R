library(tidyverse)
library(caret)
library(ROSE)
library(smotefamily)
library(Boruta)

set.seed(42)

df <- read.csv("heart_2020_cleaned.csv", stringsAsFactors = FALSE)

binary_cols <- c("HeartDisease", "Smoking", "AlcoholDrinking", "Stroke",
                 "DiffWalking", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer")
df[binary_cols] <- lapply(df[binary_cols], function(x) factor(x, levels = c("No", "Yes")))

df$Sex         <- factor(df$Sex)
df$Race        <- factor(df$Race)
df$Diabetic    <- factor(df$Diabetic, levels = c("No", "No, borderline diabetes", "Yes (during pregnancy)", "Yes"))
df$GenHealth   <- factor(df$GenHealth, levels = c("Poor", "Fair", "Good", "Very good", "Excellent"), ordered = TRUE)
df$AgeCategory <- factor(df$AgeCategory,
                         levels = c("18-24","25-29","30-34","35-39","40-44","45-49",
                                    "50-54","55-59","60-64","65-69","70-74","75-79","80 or older"),
                         ordered = TRUE)

numeric_cols <- c("BMI", "PhysicalHealth", "MentalHealth", "SleepTime")

for (col in numeric_cols) {
  lo <- quantile(df[[col]], 0.01, na.rm = TRUE)
  hi <- quantile(df[[col]], 0.99, na.rm = TRUE)
  df[[col]] <- pmin(pmax(df[[col]], lo), hi)
}

preProc_params     <- preProcess(df[, numeric_cols], method = c("center", "scale"))
df[, numeric_cols] <- predict(preProc_params, df[, numeric_cols])

dummies <- dummyVars(~ Sex + Race + Diabetic, data = df, fullRank = TRUE)
ohe_df  <- as.data.frame(predict(dummies, newdata = df))
df      <- cbind(df[ , !(names(df) %in% c("Sex", "Race", "Diabetic"))], ohe_df)
names(df) <- make.names(names(df))

df$GenHealth   <- as.integer(df$GenHealth)
df$AgeCategory <- as.integer(df$AgeCategory)

binary_feature_cols <- setdiff(binary_cols, "HeartDisease")
df[binary_feature_cols] <- lapply(df[binary_feature_cols], function(x) as.integer(x) - 1L)

train_idx <- createDataPartition(df$HeartDisease, p = 0.70, list = FALSE)
df_train  <- df[train_idx, ]
df_temp   <- df[-train_idx, ]

val_idx <- createDataPartition(df_temp$HeartDisease, p = 0.50, list = FALSE)
df_val  <- df_temp[val_idx, ]
df_test <- df_temp[-val_idx, ]

df_train_small <- df_train[sample(nrow(df_train), 1000), ]

boruta_output <- Boruta(HeartDisease ~ ., data = df_train_small, doTrace = 0, maxRuns = 11)
selected_features <- getSelectedAttributes(boruta_output, withTentative = TRUE)

df_train <- df_train[, c(selected_features, "HeartDisease")]
df_val   <- df_val[, c(selected_features, "HeartDisease")]
df_test  <- df_test[, c(selected_features, "HeartDisease")]

x_smote <- df_train[, names(df_train) != "HeartDisease"]
y_smote <- ifelse(df_train$HeartDisease == "Yes", 1, 0)

smote_out <- SMOTE(X = x_smote, target = y_smote, K = 5, dup_size = 2)
df_train_smote <- smote_out$data
names(df_train_smote)[ncol(df_train_smote)] <- "HeartDisease"
df_train_smote$HeartDisease <- factor(ifelse(df_train_smote$HeartDisease == 1, "Yes", "No"),
                                      levels = c("No", "Yes"))

df_train_rose <- ROSE(HeartDisease ~ ., data = df_train, seed = 42)$data

write.csv(df_train,       "heart_train.csv",        row.names = FALSE)
write.csv(df_train_smote, "heart_train_smote.csv",  row.names = FALSE)
write.csv(df_train_rose,  "heart_train_rose.csv",   row.names = FALSE)
write.csv(df_val,         "heart_val.csv",          row.names = FALSE)
write.csv(df_test,        "heart_test.csv",         row.names = FALSE)
write.csv(data.frame(SelectedFeatures = selected_features), "selected_features.csv", row.names = FALSE)

#----------
#The code preprocesses the dataset by first converting categorical variables, then dealing with outliers, 
#and standardising all features to make them easier for the model to use.
#It then applies one-hot encoding and splits the data into training,
#validation, and testing sets. Feature selection is performed using Boruta, 
#and class imbalance is addressed using SMOTE and ROSE, resulting in all the datasets needed for modelling and visualisation.
#--------