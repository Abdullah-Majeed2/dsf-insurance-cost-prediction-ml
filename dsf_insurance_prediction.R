
# ====================  DSF SEMESTER PROJECT  ====================
# -------  PREDICTING ANNUAL MEDICAL INSURANCE CHARGES   -------
# -------  BASED ON AGE, BMI, No OF CHILDREN, GENDER, SMOKING STATUS  -------


install.packages("rpart")
install.packages("rpart.plot")
install.packages("randomForest")


# Setting path of Dataset (use \\ in directory)
setwd("C:\\Users\\Abdullah\\Desktop\\DSF\\Semester Project")


# Loading libraries
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)


# Loading the dataset
insurance <- read.csv("insurance_data.csv")



# ====================  Data Cleaning  ====================


# Function to calculate mode
get_mode <- function(data) {

  clean_data <- data[!is.na(data)]
  value_counts <- table(clean_data)
  most_common_value <- names(value_counts)[which.max(value_counts)]
  
  return(most_common_value)
}


# Handling missing values
insurance$age[is.na(insurance$age)] <- mean(insurance$age, na.rm = TRUE)
insurance$bmi[is.na(insurance$bmi)] <- mean(insurance$bmi, na.rm = TRUE)
insurance$children[is.na(insurance$children)] <- median(insurance$children, na.rm = TRUE)
insurance$gender[is.na(insurance$gender)] <- get_mode(insurance$gender)
insurance$smoker[is.na(insurance$smoker)] <- get_mode(insurance$smoker)
insurance$region[is.na(insurance$region)] <- get_mode(insurance$region)
insurance$charges[is.na(insurance$charges)] <- median(insurance$charges, na.rm = TRUE)

 
# Removing extreme outliers
lower_cutoff <- quantile(insurance$charges,  0.025)
upper_cutoff <- quantile(insurance$charges,  0.975)

insurance <- insurance[insurance$charges >= lower_cutoff & insurance$charges <= upper_cutoff, ]


# Convert categorical variables to factors
insurance$gender <- factor(insurance$gender)
insurance$smoker <- factor(insurance$smoker)
insurance$region <- factor(insurance$region)



# ====================  Visualization (Graphs)  ====================

theme_set(theme_minimal(base_size = 15))


# Smoking Status vs Charges

ggplot(insurance, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot(outlier.color = "gray40", outlier.size = 2, width = 0.6) +
  scale_fill_manual(values = c("no" = "lightgreen", "yes" = "salmon")) +
  scale_x_discrete(labels = c("no" = "Non-Smoker", "yes" = "Smoker")) +
  labs(
    title = "Insurance Costs: Smokers vs. Non-Smokers",
    x = "", y = "Charges ($)"
  ) +
  theme(legend.position = "none")


# Age vs Charges with Smoking Status

ggplot(insurance, aes(x = age, y = charges, color = smoker)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = c("no" = "blue", "yes" = "red")) +
  labs(
    title = "Age vs Insurance Charges by Smoking Status",
    x = "Age", y = "Charges ($)",
    color = "Smoker"
  )


# BMI vs Charges with Smoking Status

ggplot(insurance, aes(x = bmi, y = charges, color = smoker)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = c("no" = "blue", "yes" = "red")) +
  labs(
    title = "BMI vs Insurance Charges by Smoking Status",
    x = "BMI", y = "Charges ($)",
    color = "Smoker"
  )


# Number of Children vs Charges

ggplot(insurance, aes(x = as.factor(children), y = charges, fill = as.factor(children))) +
  geom_boxplot(width = 0.6, outlier.color = "gray50") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "skyblue", "2" = "deepskyblue", 
                               "3" = "steelblue", "4" = "royalblue", "5" = "navy")) +
  labs(
    title = "Insurance Costs by Number of Children",
    x = "Number of Children", y = "Charges ($)",
    fill = "Children"
  )


# Distribution of Charges

ggplot(insurance, aes(x = charges)) +
  geom_histogram(fill = "skyblue", color = "white", bins = 30) +
  labs(
    title = "Distribution of Insurance Charges",
    x = "Charges ($)", y = "Frequency"
  )


# Correlation matrix

cor_data <- insurance

cor_data$gender <- as.numeric(cor_data$gender)
cor_data$smoker <- as.numeric(cor_data$smoker)
cor_data$region <- as.numeric(cor_data$region)

cor_matrix <- round(cor(cor_data), 2)
print("Correlation Matrix:")
print(cor_matrix)


# Heatmap visualization

cor_df <- as.data.frame(as.table(cor_matrix))
names(cor_df) <- c("Variable1", "Variable2", "Correlation")

ggplot(cor_df, aes(x = Variable1, y = Variable2, fill = Correlation)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) + 
  geom_text(aes(label = Correlation), size = 4.2, color = "black") +
  scale_fill_gradient2(
    low = "skyblue", high = "red", mid = "white", midpoint = 0,
    limits = c(-1, 1)
  ) +
  labs(
    title = "Correlation Heatmap",
    x = "", y = "",
    fill = "Corr"
  ) +
  theme_minimal(base_size = 15) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", color = "grey20"),
    axis.text.y = element_text(face = "bold", color = "grey20"),
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold", color = "grey11"),
    panel.grid = element_blank(),
    legend.position = "right"
  )



# ====================  Feature Engineering  ====================


# 1. Create BMI categories
insurance$bmi_category <- cut(insurance$bmi,
                              breaks = c(0, 18.5, 25, 30, Inf),
                              labels = c("Underweight", "Normal", "Overweight", "Obese"))


# 2. Age groups
insurance$age_group <- cut(insurance$age,
                           breaks = c(17, 30, 40, 50, 60, 70),
                           labels = c("18-29", "30-39", "40-49", "50-59", "60+"))


# 3. Smoker-age interaction
insurance$smoker_age <- insurance$age * (insurance$smoker == "yes")


# 4. Smoker-BMI interaction
insurance$smoker_bmi <- insurance$bmi * (insurance$smoker == "yes")


# 5. Children binary (has children or not)
insurance$has_children <- ifelse(insurance$children > 0, 1, 0)


# ====================  Training Model  ==================== 

# Splitting training data and testing data
set.seed(123)
train_index <- sample(1:nrow(insurance), 0.8 * nrow(insurance))
train_data <- insurance[train_index, ]
test_data <- insurance[-train_index, ]


# 1. Linear Regression
lm_model <- lm(charges ~ age + bmi + children + gender + region + smoker + 
                smoker_age + smoker_bmi + has_children,
                data = train_data)



# 2. Decision Tree
tree_model <- rpart(charges ~ age + bmi + children + gender + region + smoker + 
                      smoker_age + smoker_bmi + has_children, 
                    data = train_data,
                    method = "anova",
                    control = rpart.control(cp = 0.005, minsplit = 15)
                    )

rpart.plot(tree_model, box.palette = "BuBn",
           main = "Insurance Cost Decision Tree", yesno = 2,)

        

# 3. Random Forest
rf_model <- randomForest(charges ~ age + bmi + children + gender + region + smoker + 
                           smoker_age + smoker_bmi + has_children,
                         data = train_data,
                         ntree = 250,
                         importance = TRUE)

varImpPlot(rf_model, main = "Random Forest Variable Importance")



# ====================  Model Comparison and Evaluation  ====================

# Make predictions
test_data$lm_pred <- predict(lm_model, test_data)
test_data$tree_pred <- predict(tree_model, test_data)
test_data$rf_pred <- predict(rf_model, test_data)


# Function to calculate evaluation metrics
calculate_metrics <- function(actual, predicted) {
  mae <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))
  mape <- mean(abs((actual - predicted)/actual)) * 100
  
  
  return(c( MAE = mae, RMSE = rmse, MAPE = mape))
}


# Calculate metrics
metrics <- data.frame(
  Model = c("Linear Regression", "Decision Tree", "Random Forest"),
  MAE = round(c(
    calculate_metrics(test_data$charges, test_data$lm_pred)["MAE"],
    calculate_metrics(test_data$charges, test_data$tree_pred)["MAE"],
    calculate_metrics(test_data$charges, test_data$rf_pred)["MAE"]
  ), 2),
  RMSE = round(c(
    calculate_metrics(test_data$charges, test_data$lm_pred)["RMSE"],
    calculate_metrics(test_data$charges, test_data$tree_pred)["RMSE"],
    calculate_metrics(test_data$charges, test_data$rf_pred)["RMSE"]
  ), 2),
  MAPE = round(c(
    calculate_metrics(test_data$charges, test_data$lm_pred)["MAPE"],
    calculate_metrics(test_data$charges, test_data$tree_pred)["MAPE"],
    calculate_metrics(test_data$charges, test_data$rf_pred)["MAPE"]
  ), 2)
)


print("Model Performance :")
print(metrics)



# ====================  Visualization for Actual Vs Predicted Values  ====================

# Actual vs Predicted for Linear Regression
ggplot(test_data, aes(x = charges, y = lm_pred, color = smoker)) +
  geom_point(alpha = 0.8, size = 2.2) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", linewidth = 1.2) +
  scale_color_manual(values = c("no" = "blue", "yes" = "red")) +
  labs(
    title = "Actual vs Predicted Charges (Linear Regression)",
    x = "Actual Charges ($)",
    y = "Predicted Charges ($)",
    color = "Smoker"
  )


# Actual vs Predicted for Decision Tree
ggplot(test_data, aes(x = charges, y = tree_pred, color = smoker)) +
  geom_point(alpha = 0.8, size = 2.2) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", linewidth = 1.2) +
  scale_color_manual(values = c("no" = "blue", "yes" = "red")) +
  labs(
    title = "Actual vs Predicted Charges (Decision Tree)",
    x = "Actual Charges ($)",
    y = "Predicted Charges ($)",
    color = "Smoker"
  )


# Actual vs Predicted for Random Forest
ggplot(test_data, aes(x = charges, y = rf_pred, color = smoker)) +
  geom_point(alpha = 0.8, size = 2.2) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", linewidth = 1.2) +
  scale_color_manual(values = c("no" = "blue", "yes" = "red")) +
  labs(
    title = "Actual vs Predicted Charges (Random Forest)",
    x = "Actual Charges ($)",
    y = "Predicted Charges ($)",
    color = "Smoker"
  )



# ====================  User Data Prediction  ====================

# Function to convert user data to required data type

predict_insurance_cost <- function(age, bmi, children, gender, region, smoker) {
  # Creating data frame
  new_data <- data.frame(
    age = age,
    bmi = bmi,
    children = children,
    gender = factor(gender, levels = levels(train_data$gender)),
    region = factor(region, levels = levels(train_data$region)),
    smoker = factor(smoker, levels = levels(train_data$smoker)),
    smoker_age = age * (smoker == "yes"),
    smoker_bmi = bmi * (smoker == "yes"),
    has_children = ifelse(children > 0, 1, 0)
  )
  
  # predictions
  predictions <- data.frame(
    Model = c("Linear Regression", "Decision Tree", "Random Forest"),
    Predicted_Cost = c(
      predict(lm_model, new_data),
      predict(tree_model, new_data),
      predict(rf_model, new_data)
    )
  )
  
  return(predictions)
}

# Making predictions

user_data_pred <- predict_insurance_cost(
  age = 21,
  gender = "male",
  bmi = 35.53,
  children = 0,
  smoker = "no",
  region = "southeast"
)

print("Predicted Value for this client: ")
print(user_data_pred)


# ====================  THE END  ====================




