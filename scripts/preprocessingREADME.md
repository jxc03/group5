The code preprocesses the dataset by first converting categorical variables, then dealing with outliers, and standardising all features to make them easier for the model to use. It then applies one-hot encoding and splits the data into training, validation, and testing sets. Feature selection is performed using Boruta, and class imbalance is addressed using SMOTE and ROSE, resulting in all the datasets needed for modelling and visualisation.

- heart_train.csv → this is the main training dataset (used to actually train the models)
- heart_train_smote.csv → same as training but balanced using SMOTE (creates synthetic minority cases so the model doesn’t ignore heart disease cases)
- heart_train_rose.csv → another balanced version using ROSE (different method, good for comparison)
- heart_val.csv → validation set (used to tune/check models before final testing)
- heart_test.csv → final test set (completely untouched, used to evaluate how well the model actually performs)
- selected_features.csv → features picked by Boruta (basically the columns that actually matter for predicting heart disease)

The idea is:
- train models on original + SMOTE + ROSE
- compare performance
- always evaluate on the same test set so results are fair