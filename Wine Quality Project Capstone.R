#################################################
# Wine Quality Prediction Capstone 
################################################

# 1. Introduction

# For this project, we will create a model to predict the wine quality based on a set of 11 attributes 
# detailing the physicochemical properties of a selection of wines from the north of Portugal.

# 2. Methodology and Analysis

## 2.1. Data import and pre-processing

# Install the required libraries if they have not been installed yet
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# Load the required libraries
library(tidyverse)
library(caret)
library(data.table)
library(kernlab)   
library(e1071)
library(randomForest)
library(corrplot)
library(gridExtra)
library(purrr)

# Load the 2 csv files and store the date in 2 corresponding variables
red_wines <- read.csv2("./wine-quality-dataset/winequality-red.csv")
white_wines <- read.csv2("./wine-quality-dataset/winequality-white.csv")
# Display the dimensions of the 2 dataset tables
dim(red_wines)
dim(white_wines)

# Display the first lines of the red wines dataset
head(red_wines)

# Format the variable columns that were in character as numeric values
red_wines <- red_wines %>% mutate_if(is.character,as.numeric)
# Format the quality column that was stored as integer as factors
white_wines <- white_wines %>% mutate_if(is.character,as.numeric)
#Display the the first lines of the red wines dataset to check the new column format
head(red_wines)

# Check if the red wine dataset or the white wine dataset have any missing value
any(is.na(red_wines))
any(is.na(white_wines))

# Merge the red wine and white wine datasets
wines <- rbind(red_wines,white_wines)
#Display the dimension of the new wines dataset
dim(wines)

## 2.2. Data Exploration and Preparation

### 2.2.1. The wine quality

# Display the histogram of the wine quality
hist(wines$quality)

# Create a 3 classes quality based on the above rating ranges
quality_3_classes=ifelse(wines$quality<=4,3,ifelse((wines$quality<7),2,1))
# Display the bar plot of the proportion of wine ratings with this 3 classes quality
barplot(prop.table(table(quality_3_classes))*100)

# Display the percentage of ratings per category
round(prop.table(table(quality_3_classes))*100,1)

# Create a 2 classes quality based on the above rating ranges
quality_2_classes = ifelse(wines$quality<=6,0,1)
# Display the bar plot of the proportion of wine ratings with this 2 classes quality
barplot(prop.table(table(quality_2_classes))*100,1)

### 2.2.2. The attributes

# Create an histogram for the different attributes
wines %>%
  gather(Attributes, value, 1:11) %>% # Transform the table to create the chart
  ggplot(aes(x=value, fill=Attributes)) + # Create a grid of the attributes histograms
  geom_histogram(colour="black", show.legend=FALSE) +
  facet_wrap(~Attributes, scales="free_x") +
  labs(x="Values", y="Frequency", title="Wines Attributes - Histograms") 

# Create a correlation plot of the attributes
corrplot(cor(wines), type="upper", method="pie", tl.cex=0.8)

# Create the wine dataset for modeling
wines_base <- wines[1:11] %>% mutate(quality = as.factor(quality_2_classes))
# Display the first rows of the dataset
head(wines_base)

## 2.3. Modeling

### 2.3.1. Create the training and test sets

# Split the dataset in a training set and a test set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = wines_base$quality, times = 1, p = 0.1, list = FALSE)
train_set_orig <- wines_base[-test_index,]
test_set_orig <- wines_base[test_index,]

# Create a standardisation process based on the training set values
stand_process <- preProcess(train_set_orig, method = c("center", "scale"))

# Apply this standardisation process to the training and test sets
train_set_full <- predict(stand_process, train_set_orig)
test_set <- predict(stand_process, test_set_orig)

# Split the full training set in a training set and a validation set
set.seed(1, sample.kind="Rounding")
validation_index <- createDataPartition(y = train_set_full$quality, times = 1, p = 0.1, list = FALSE)
train_set <-train_set_full[-validation_index,]
validation_set <- train_set_full[validation_index,]

### 2.3.2. Support Vector Machine (SVM)

#### Choice of the kernel

# Set the kernels list that we are going to test
kernels=c('svmLinear', 'svmPoly','svmRadial','svmRadialSigma')

# Create a empty table that we are going to use to store the accuracy of the model with the different kernel
results_svm_kernel <- tibble(Method=NULL, Hyperparameters = NULL, Accuracy = NULL, Balanced_Accuracy = NULL, Sensitivity = NULL, Specificity = NULL, F1_Score = NULL)

# Train the SVM model with the different kernels and store the corresponding accuracy in the table
accuracies <- sapply(kernels, function(k){
  set.seed(1, sample.kind="Rounding")
  model <- train(
    quality~.,data=train_set_full, method = k,
    trControl = trainControl("cv", number = 10))
  acc_train <- max(model$results$Accuracy)
  results_svm_kernel <- bind_rows(results_svm_kernel,tibble(Method="SVM", Accuracy_training = acc_train))
})

# Set the number of digits for pdf print
options(pillar.sigfig = 7)

# Display the accuracy results for the different kernels
t(accuracies)

#### Training and optimizing through cross validation with tuneLength

# Train the SVM model with the kernel "svmRadialSigma" and the parameter tuneLength
set.seed(1, sample.kind="Rounding")
model_svm1 <- train(
  quality~.,data=train_set_full, method = 'svmRadialSigma',
  trControl = trainControl("cv", number = 10),
  tuneLength = 10)

# Create a chart of the accuracies for the different combinations tested
ggplot(model_svm1, highlight = TRUE)

# Display the values of the best combination
model_svm1$bestTune

# Store the accuracy of the best combination
acc_train_svm1 <- max(model_svm1$results$Accuracy)

# Create the results table for the model
results_svm <-tibble(Method="SVM", Hyperparameters="sigma=0.1434091, C=128", "Accuracy Cross Validation" = acc_train_svm1)

#### Training and optimizing through cross validation with a set values of hyperparameters

# Train the SVM model with the kernel "svmRadialSigma" and a pre-defined selection of hyperparameters
set.seed(1, sample.kind="Rounding")
model_svm2 <- train(
  quality~.,data=train_set_full, method = "svmRadialSigma",
  trControl = trainControl("cv", number = 10),
  tuneGrid =  expand.grid(sigma=c(0.2,0.35,0.4,0.5),C=c(1,25,50,100,128))
)

# Create a chart of the accuracies for the different combinations tested
ggplot(model_svm2, highlight = TRUE)

# Display the values of the best combination
model_svm2$bestTune

# Store the accuracy of the best combination
acc_train_svm2 <- max(model_svm2$results$Accuracy)

# Update the results table for the model
results_svm2 <- tibble(Method="SVM", Hyperparameters="sigma=0.5, C=128", "Accuracy Cross Validation" = acc_train_svm2)
results_svm <- bind_rows(results_svm,results_svm2)

### 2.3.3. k-Nearest Neighbors (kNN)

#### Training and optimizing through cross validation with tuneLength

# Train the model with the parameter tuneLength=10
set.seed(1, sample.kind="Rounding")
model_knn1 <- train(
  quality~.,data=train_set_full, method="knn",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10)

# Plot model accuracy vs different values of k
ggplot(model_knn1,highlight = TRUE)

# Display the values of the best combination
model_knn1$bestTune

# Store the accuracy of the best combination
acc_train_knn1 <- max(model_knn1$results$Accuracy)

# Create the results table for the model
results_kNN <- tibble(Method="kNN", Hyperparameters = "k=7", "Accuracy Cross Validation" = acc_train_knn1)
results_kNN

#### Training and optimizing through cross validation with q set values of hyperparameters

# Train the model with a pre-defined selection of k values
set.seed(1, sample.kind="Rounding")
model_knn2 <- train(
  quality~.,data=train_set_full, method="knn",
  trControl =  trainControl(method = "cv", number = 10),
  tuneGrid = data.frame(k = seq(3, 71, 2))
)
# Plot model accuracy vs different values of Cost
plot(model_knn2)

# Display the values of the best combination
model_knn2$bestTune

# Store the accuracy of the best combination
acc_train_knn2 <- max(model_knn2$results$Accuracy)

# Update the results table for the model
results_kNN2 <- tibble(Method="kNN", Hyperparameters = "k=3", "Accuracy Cross Validation" = acc_train_knn2)
results_kNN <- bind_rows(results_kNN,results_kNN2)

## 2.3.4. Random Forest

#### Training and optimizing through cross validation with tuneLength

# Train the model with the parameter tuneLength=10
set.seed(1, sample.kind="Rounding")
model_rf1 <- train(
  quality~.,data=train_set_full, method="rf",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10)

# Plot model accuracy vs different values of mtry
plot(model_rf1)

# Display the parameters of the best combination
model_rf1$finalModel

# Store the accuracy of the best combination
acc_train_rf1 <- max(model_rf1$results$Accuracy)

# Create the results table for the model
results_rf <- tibble(Method="Random Forest", Hyperparameters = "ntree=500, mtry=8", "Accuracy Cross Validation" = acc_train_rf1)

#### Training and optimizing on the validation set with a set values of hyperparameters

# Set the ntree values we want to test 
ntree=seq(50,600,50)

# Test the Random Forest model with the different values of ntree on the validation test and store the corresponding accuracy
acc <- sapply(ntree, function(nt){
  set.seed(1, sample.kind="Rounding")
  model <- randomForest(quality~.,data=train_set,mtry=6,ntree=nt)
  predict_quality <- predict(model,validation_set[1:11])
  cm <- confusionMatrix(predict_quality,validation_set$quality)
  cm$overall["Accuracy"]
})

# Plot the accuracy of the model vs the corresponding number of trees
qplot(ntree, acc)

# Select the first value of ntree that provides the maximum accuracy
ntree[which.max(acc)]

# Train the model with the parameters mtry=8 and ntree = 100
set.seed(1, sample.kind="Rounding")
model_rf3 <- randomForest(quality~.,data=train_set_full,mtry=8,ntree=100)

# Store the corresponding accuracy
acc_train_rf1 <- max(acc)

# Update the results table for the model
results_rf3 <- tibble(Method="Random Forest", Hyperparameters = "ntree=100, mtry=8", "Accuracy Cross Validation" = acc_train_rf1)
results_rf<- bind_rows(results_rf,results_rf3)

# 3. Results

## 3.1. SVM

# Fit the model on the full training set with the hyperparameters selected in the modeling phase
set.seed(1, sample.kind="Rounding")
model_svm <- train(
  quality~.,data=train_set_full, method = "svmRadialSigma",
  tuneGrid =  expand.grid(sigma=0.5,C=128)
)

# Store the predictions based on the test set
predict_quality <- predict(model_svm,test_set[1:11])

# Compute and store the confusion matrix
cm <- confusionMatrix(predict_quality,test_set$quality, positive = "1")

# Create results table to compare the different model performances
results <- tibble(Method="SVM", Hyperparameters="sigma=0.5, C=128", Accuracy = cm$overall["Accuracy"], Balanced_Accuracy = cm$byClass["Balanced Accuracy"], Sensitivity = cm$byClass["Sensitivity"], Specificity = cm$byClass["Specificity"], F1_Score = F_meas(predict_quality,test_set$quality))

## 3.2. kNN

# Fit the model on the full training set with the hyperparameters selected in the modeling phase
set.seed(1, sample.kind="Rounding")
model_knn <- train(
  quality~.,data=train_set_full, method = "knn",
  tuneGrid =  expand.grid(k=3)
)

# Store the predictions based on the test set
predict_quality <- predict(model_knn,test_set[1:11])

# Compute and store the confusion matrix
cm <- confusionMatrix(predict_quality,test_set$quality,positive = "1")

# Update the results table
result_kNN <- tibble(Method="kNN", Hyperparameters="k=3", Accuracy = cm$overall["Accuracy"], Balanced_Accuracy = cm$byClass["Balanced Accuracy"], Sensitivity = cm$byClass["Sensitivity"], Specificity = cm$byClass["Specificity"], F1_Score = F_meas(predict_quality,test_set$quality))
results <- bind_rows(results,result_kNN)

## 3.3. Random Forest

# Fit the model on the full training set with the hyperparameters selected in the modeling phase
set.seed(1, sample.kind="Rounding")
model_rf <- randomForest(quality~.,data=train_set_full,ntree=100, mtry=8)

# Store the predictions based on the test set
predict_quality <- predict(model_rf,test_set[1:11])

# Compute and store the confusion matrix
cm <- confusionMatrix(predict_quality,test_set$quality, positive = "1")

# Update the results table
result_rf <- tibble(Method="Random Forest", Hyperparameters="ntree=100, mtry=8", Accuracy = cm$overall["Accuracy"], Balanced_Accuracy = cm$byClass["Balanced Accuracy"], Sensitivity = cm$byClass["Sensitivity"], Specificity = cm$byClass["Specificity"], F1_Score = F_meas(predict_quality,test_set$quality))
results <- bind_rows(results,result_rf)

## 3.4. Logistic Regression

# Fit the model on the full training set
set.seed(1, sample.kind="Rounding")
model_lg <- train(quality~.,data=train_set_full, method = 'glm')

# Store the predictions based on the test set
predict_quality <- predict(model_lg,test_set[1:11])

# Compute and store the confusion matrix
cm <- confusionMatrix(predict_quality,test_set$quality, positive = "1")

# Create the results table for the model
result_lg <- tibble(Method="Logistic Regression", Hyperparameters="NA",  Accuracy = cm$overall["Accuracy"], Balanced_Accuracy = cm$byClass["Balanced Accuracy"], Sensitivity = cm$byClass["Sensitivity"], Specificity = cm$byClass["Specificity"],F1_Score = F_meas(predict_quality,test_set$quality))
results <- bind_rows(results,result_lg) 

## 3.5. Performance analysis

# Display the performance results
results

# Create a plot to represent the performance of the models on the different indicators
results %>%
  subset(select=-Hyperparameters) %>%
  gather(Indicator, Value, Accuracy:F1_Score) %>%
  ggplot(aes(x=Indicator, y=Value, col=Method)) +
  geom_point()




