install.packages("tidyverse")
library(tidyverse)
install.packages("janitor")
library(janitor)

df <- read.csv("data/heart_2020_cleaned.csv", stringsAsFactors = FALSE)

df <- clean_names(df)

names(df)
str(df)
getwd()
file.exists("data/heart_2020_cleaned.csv")
df <- read.csv("data/heart_2020_cleaned.csv")
str(df)
table(df$heart_disease)
prop.table(table(df$heart_disease)) * 100
names(df)
table(df$HeartDisease)
prop.table(table(df$HeartDisease)) * 100
ggplot(df, aes(x = HeartDisease)) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Distribution of Heart Disease",
    x = "Heart Disease",
    y = "Count"
  )
ggsave("outputs/plots/heart_disease_distribution.png", width = 7, height = 5)
ggplot(df, aes(x = HeartDisease, y = BMI)) +
  geom_boxplot(fill = "lightblue") +
  labs(
    title = "BMI by Heart Disease Status",
    x = "Heart Disease",
    y = "BMI"
  )
ggsave("outputs/plots/heart_disease_distribution.png", width = 7, height = 5)
p1 <- ggplot(df, aes(x = HeartDisease)) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Distribution of Heart Disease",
    x = "Heart Disease",
    y = "Count"
  )

p1

ggsave("outputs/plots/heart_disease_distribution.png", plot = p1, width = 7, height = 5)
p2 <- ggplot(df, aes(x = HeartDisease, y = BMI)) +
  geom_boxplot(fill = "lightblue") +
  labs(
    title = "BMI by Heart Disease Status",
    x = "Heart Disease",
    y = "BMI"
  )

p2

ggsave("outputs/plots/bmi_vs_heart_disease.png", plot = p2, width = 7, height = 5)
p3 <- ggplot(df, aes(x = AgeCategory, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  labs(
    title = "Heart Disease by Age Category",
    x = "Age Category",
    y = "Proportion"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p3

ggsave("outputs/plots/age_vs_heart_disease.png", plot = p3, width = 8, height = 5)
p4 <- ggplot(df, aes(x = Smoking, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  labs(
    title = "Heart Disease by Smoking Status",
    x = "Smoking",
    y = "Proportion"
  )

p4

ggsave("outputs/plots/smoking_vs_heart_disease.png", plot = p4, width = 7, height = 5)
p5 <- ggplot(df, aes(x = GenHealth, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  labs(
    title = "Heart Disease by General Health",
    x = "General Health",
    y = "Proportion"
  )

p5

ggsave("outputs/plots/genhealth_vs_heart_disease.png", plot = p5, width = 7, height = 5)
p6 <- ggplot(df, aes(x = Sex, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  labs(
    title = "Heart Disease by Sex",
    x = "Sex",
    y = "Proportion"
  )

p6

ggsave("outputs/plots/sex_vs_heart_disease.png", plot = p6, width = 7, height = 5)
df %>%
  group_by(HeartDisease) %>%
  summarise(
    mean_bmi = mean(BMI, na.rm = TRUE),
    sd_bmi = sd(BMI, na.rm = TRUE),
    median_bmi = median(BMI, na.rm = TRUE)
  )
prop.table(table(df$Smoking, df$HeartDisease), margin = 1)
prop.table(table(df$Sex, df$HeartDisease), margin = 1)
prop.table(table(df$GenHealth, df$HeartDisease), margin = 1)
t.test(BMI ~ HeartDisease, data = df)
chisq.test(table(df$Smoking, df$HeartDisease))
chisq.test(table(df$Sex, df$HeartDisease))
chisq.test(table(df$GenHealth, df$HeartDisease))
write.csv(df, "outputs/results/cleaned_data.csv")
