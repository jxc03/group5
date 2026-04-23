library(tidyverse)
install.packages("caret")
library(caret)
install.packages("randomForest")
install.packages("pROC")
library(randomForest)
library(pROC)

# Load dataset
df <- read.csv("data/heart_2020_cleaned.csv", stringsAsFactors = FALSE)

# Convert character columns to factors
df[] <- lapply(df, function(x) {
  if (is.character(x)) as.factor(x) else x
})

# Convert target variable
df$HeartDisease <- as.factor(df$HeartDisease)

# Split data
set.seed(123)

train_index <- createDataPartition(df$HeartDisease, p = 0.7, list = FALSE)
train_data <- df[train_index, ]
test_data  <- df[-train_index, ]

# Check sizes
dim(train_data)
dim(test_data)
rf_model <- randomForest(
  HeartDisease ~ .,
  data = train_data,
  ntree = 200,
  importance = TRUE
)

print(rf_model)
rf_pred_class <- predict(rf_model, newdata = test_data, type = "response")
rf_pred_prob  <- predict(rf_model, newdata = test_data, type = "prob")[, "Yes"]
cm <- confusionMatrix(rf_pred_class, test_data$HeartDisease, positive = "Yes")
print(cm)
roc_obj <- roc(test_data$HeartDisease, rf_pred_prob, levels = c("No", "Yes"))
auc(roc_obj)
png("outputs/plots/rf_roc_curve.png", width = 800, height = 600)
plot(roc_obj, main = "ROC Curve - Random Forest")
dev.off()
png("outputs/plots/rf_feature_importance.png", width = 900, height = 700)
varImpPlot(rf_model, main = "Random Forest Feature Importance")
dev.off()
plot(roc_obj, main = "ROC Curve - Random Forest")
head(rf_pred_class)
head(rf_pred_prob)
library(pROC)

roc_obj <- roc(test_data$HeartDisease, rf_pred_prob, levels = c("No", "Yes"))
print(roc_obj)
plot(roc_obj, main = "ROC Curve - Random Forest")
png("outputs/plots/rf_roc_curve.png", width = 800, height = 600)
plot(roc_obj, main = "ROC Curve - Random Forest")
dev.off()
png("outputs/plots/rf_feature_importance.png", width = 900, height = 700)

varImpPlot(rf_model, main = "Random Forest Feature Importance")

dev.off()
# Confusion matrix
cm <- confusionMatrix(rf_pred_class, test_data$HeartDisease, positive = "Yes")
print(cm)

# AUC
auc_value <- auc(roc_obj)
print(auc_value)

# F1-score
precision <- cm$byClass["Pos Pred Value"]
recall <- cm$byClass["Sensitivity"]
f1 <- 2 * ((precision * recall) / (precision + recall))

# Save metrics into a small table
rf_results <- data.frame(
  Model = "Random Forest",
  Accuracy = cm$overall["Accuracy"],
  Balanced_Accuracy = cm$byClass["Balanced Accuracy"],
  Precision = precision,
  Recall = recall,
  Specificity = cm$byClass["Specificity"],
  F1_Score = f1,
  ROC_AUC = as.numeric(auc_value)
)

print(rf_results)

write.csv(rf_results, "outputs/results/random_forest_metrics.csv", row.names = FALSE)
importance(rf_model)
rf_importance <- as.data.frame(importance(rf_model))
rf_importance$Feature <- rownames(rf_importance)

rf_importance_sorted <- rf_importance[order(-rf_importance$MeanDecreaseGini), ]

head(rf_importance_sorted, 10)

write.csv(rf_importance_sorted, "outputs/results/random_forest_feature_importance.csv", row.names = FALSE)
