print(getwd())
setwd("C:/Users/Tacha/Documents/MAIA/SECOND SEMESTER/Statistical Learning/Final")
print(getwd())
library(ISLR2)
library(leaps) # for regsubsets
library(glmnet)
library(caret)
library(MASS)
library(boot)
library(pROC)
library(mltools)

# read data
data <- read.csv("ADCTLtrain.csv")
data <- data[,-1]
attach(data)

# data split
cutoff = round(0.8*nrow(data))
data_train <- data[1:cutoff,]
data_val <- data[-(1:cutoff),]

data_train_y = data_train[,length(data_train)]
data_test_y = data_val[,length(data_val)]
data_final_y = data[,length(data_val)]

data_train_y_enc = rep(0,nrow(data_train))
data_train_y_enc[data_train_y == 'AD'] = 1
data_test_y_enc = rep(0,nrow(data_val))
data_test_y_enc[data_test_y == 'AD'] = 1
data_final_y_enc = rep(0,nrow(data))
data_final_y_enc[data_final_y == 'AD'] = 1

#################### FEATURE SELECTION ########################
# lasso
lambda <- 10^seq(10, -2, length = 100)
set.seed(1)
lasso.mod = train(
  Label ~., data = data, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda)
)
lasso.mod$bestTune$lambda

# Get the relevant features and their values
features = data.frame(as.matrix(coef(lasso.mod$finalModel, lasso.mod$bestTune$lambda)))
features$names = rownames(features)
features = features[features$s1 != 0,]
features = features[-2]
features_names = rownames(features)
features_names = features_names[-1]

sel_columns = colnames(data) %in% features_names
sel_columns[length(sel_columns)] = TRUE

##### Filtered data
data_filtered = data_train[, sel_columns]
data_filtered_test = data_val[, sel_columns]

data_filtered_final = data[, sel_columns]

##### set control for CV
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

################# LDA ################
set.seed(1)
lda.model <- train(Label ~ ., data = data_filtered, method = "lda", trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
print(lda.model) # ROC 0.984

################ QDA #####################
set.seed(1)
qda.model <- train(Label ~ ., data = data_filtered, method = "qda", trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
print(qda.model) # ROC 0.887

################ LOGISTIC REGRESSION #################
set.seed(1)
glm.model <- train(Label ~ ., data = data_filtered, method = "glm", trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
print(glm.model) # Too many predictors for this model, so we discard it even though ROC = 0.995

################ NAIVE BAYES ####################
set.seed(1)
nb_grid <- expand.grid(usekernel = c(TRUE, FALSE), laplace = c(0, 0.5, 1), adjust = c(0.75, 1, 1.25, 1.5))
nb.model <- train(Label ~ ., data = data_filtered, method = "naive_bayes", usepoisson = TRUE, tuneGrid = nb_grid,  trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
nb.model$finalModel$tuneValue
print(nb.model) # ROC  0.9655


############## LDA HAS THE HIGHEST AUC SO...
pred_prob_train <- predict(lda.model, data_filtered, type="prob")
pred_prob_val <- predict(lda.model, data_filtered_test, type="prob")

pred_class_train = predict(lda.model, data_filtered)
pred_class_test = predict(lda.model, data_filtered_test)

auc(data_train_y_enc,pred_prob_train[,1]) # 0.9998
auc(data_test_y_enc,pred_prob_val[,1]) # 0.9925

mcc(pred_class_train,as.factor(data_train_y)) # 0.985
mcc(pred_class_test,as.factor(data_test_y)) # 0.83
# Therefore, we can conclude that our model did not overfit, as the mcc and the auc values are quite high and close to the train values

################################## FINAL MODEL ###################################
ctrl_final = trainControl(method = "none", classProbs = TRUE, summaryFunction = twoClassSummary)

# LDA MODEL WITH 38 VARIABLES ON WHOLE DATASET
lda.modelfinal = train(Label ~., data = data_filtered_final, method = "lda", preProcess=c("center","scale"), trControl = ctrl_final, metric = 'ROC')

# get metrics from training set
pred_prob_train = predict(lda.modelfinal, newdata = data_filtered_final, type = "prob")
pred_class_train = predict(lda.modelfinal, newdata = data_filtered_final)
auc(data_final_y_enc,pred_prob_train[,1]) # 1
mcc(pred_class_train,as.factor(data_final_y)) # 0.987

# do prediction on test_set
data_test = read.csv('ADCTLtest.csv')
data_id = data_test[,1]
data_test = data_test[,-1]

pred_prob_test = predict(lda.modelfinal, newdata = data_test, type = "prob")
pred_class_test = predict(lda.modelfinal, newdata = data_test)

ADCTLresults = data.frame(Id = data_id, Labels = pred_class_test, Probs = pred_prob_test[,1])
save(ADCTLresults, file = '0068100_Sam_ADCTLres.Rdata')

#get column indexes
Columns_index = which(names(data)%in%features_names) 
save(Columns_index, file = '0068100_Sam_ADCTLfeat.Rdata')

ROC_models = c(0.984, 0.887, 0.995, 0.9655)
plot(ROC_models, type="b", xlab = "Model", ylab="AUC", las = 2, xaxt = "n", main="10-fold CV AUC for each model")
grid()
axis(1, at=1:4, labels=c("LDA", "QDA", "Log Reg", "Naive Bayes"))


########################### SECOND EXERCISE ###################################
# read data
data <- read.csv("ADMCItrain.csv")
data <- data[,-1]
attach(data)

# data split
cutoff = round(0.8*nrow(data))
data_train <- data[1:cutoff,]
data_test <- data[-(1:cutoff),]

data_train_y = data_train[,length(data_train)]
data_test_y = data_test[,length(data_test)]
data_final_y = data[,length(data_test)]

data_train_y_enc = rep(0,nrow(data_train))
data_train_y_enc[data_train_y == 'AD'] = 1
data_test_y_enc = rep(0,nrow(data_test))
data_test_y_enc[data_test_y == 'AD'] = 1
data_final_y_enc = rep(0,nrow(data))
data_final_y_enc[data_final_y == 'AD'] = 1

#################### FEATURE SELECTION ########################
# lasso
lambda <- 10^seq(10, -2, length = 100)
set.seed(1)
lasso.mod = train(
  Label ~., data = data, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda)
)
lasso.mod$bestTune$lambda

# Get the relevant features and their values
features = data.frame(as.matrix(coef(lasso.mod$finalModel, lasso.mod$bestTune$lambda)))
features$names = rownames(features)
features = features[features$s1 != 0,]
features = features[-2]
features_names = rownames(features)
features_names = features_names[-1]

sel_columns = colnames(data) %in% features_names
sel_columns[length(sel_columns)] = TRUE

##### Filtered data
data_filtered = data_train[, sel_columns]
data_filtered_test = data_test[, sel_columns]

data_filtered_final = data[, sel_columns]

##### set control for CV
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

################# LDA ################
set.seed(1)
lda.model <- train(Label ~ ., data = data_filtered, method = "lda", trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
print(lda.model) # ROC 0.800

################ QDA #####################
set.seed(1)
qda.model <- train(Label ~ ., data = data_filtered, method = "qda", trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
print(qda.model) # ROC 0.706

################ LOGISTIC REGRESSION #################
set.seed(1)
glm.model <- train(Label ~ ., data = data_filtered, method = "glm", trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
print(glm.model) # ROC = 0.804

################ NAIVE BAYES ####################
set.seed(1)
nb_grid <- expand.grid(usekernel = c(TRUE, FALSE), laplace = c(0, 0.5, 1), adjust = c(0.75, 1, 1.25, 1.5))
nb.model <- train(Label ~ ., data = data_filtered, method = "naive_bayes", usepoisson = TRUE, tuneGrid = nb_grid,  trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
nb.model$finalModel$tuneValue
print(nb.model) # ROC  0.792


############## LOGREG HAS THE HIGHEST AUC SO...
pred_prob_train <- predict(glm.model, data_filtered, type="prob")
pred_prob_val <- predict(glm.model, data_filtered_test, type="prob")

pred_class_train = predict(glm.model, data_filtered)
pred_class_test = predict(glm.model, data_filtered_test)

auc(data_train_y_enc,pred_prob_train[,1]) # 0.8746
auc(data_test_y_enc,pred_prob_val[,1]) # 0.92

mcc(pred_class_train,as.factor(data_train_y)) # 0.6506
mcc(pred_class_test,as.factor(data_test_y)) # 0.5825
# Therefore, we can conclude that our model did not overfit, as the mcc and the auc values are quite high and close to the train values

################################## FINAL MODEL ###################################
ctrl_final = trainControl(method = "none", classProbs = TRUE, summaryFunction = twoClassSummary)

# LDA MODEL WITH 38 VARIABLES ON WHOLE DATASET
glm.modelfinal = train(Label ~., data = data_filtered_final, method = "glm", preProcess=c("center","scale"), trControl = ctrl_final, metric = 'ROC')

# get metrics from training set
pred_prob_train = predict(glm.modelfinal, newdata = data_filtered_final, type = "prob")
pred_class_train = predict(glm.modelfinal, newdata = data_filtered_final)
auc(data_final_y_enc,pred_prob_train[,1]) # 0.8837
mcc(pred_class_train,as.factor(data_final_y)) # 0.5918

# do prediction on test_set
data_test = read.csv('ADMCItest.csv')
data_id = data_test[,1]
data_test = data_test[,-1]

pred_prob_test = predict(glm.modelfinal, newdata = data_test, type = "prob")
pred_class_test = predict(glm.modelfinal, newdata = data_test)

ADMCIresults = data.frame(Id = data_id, Labels = pred_class_test, Probs = pred_prob_test[,1])
save(ADMCIresults, file = '0068100_Sam_ADMCIres.Rdata')

#get column indexes
Columns_index = which(names(data)%in%features_names) 
save(Columns_index, file = '0068100_Sam_ADMCIfeat.Rdata')


ROC_models = c(0.800, 0.706, 0.804, 0.792)
plot(ROC_models, type="b", xlab = "Model", ylab="AUC", las = 2, xaxt = "n", main="10-fold CV AUC for each model")
grid()
axis(1, at=1:4, labels=c("LDA", "QDA", "Log Reg", "Naive Bayes"))




####################### THIRD EXERCISE ##################################
# read data
data <- read.csv("MCICTLtrain.csv")
data <- data[,-1]
attach(data)

# data split
cutoff = round(0.8*nrow(data))
data_train <- data[1:cutoff,]
data_test <- data[-(1:cutoff),]

data_train_y = data_train[,length(data_train)]
data_test_y = data_test[,length(data_test)]
data_final_y = data[,length(data_test)]

data_train_y_enc = rep(0,nrow(data_train))
data_train_y_enc[data_train_y == 'MCI'] = 1
data_test_y_enc = rep(0,nrow(data_test))
data_test_y_enc[data_test_y == 'MCI'] = 1
data_final_y_enc = rep(0,nrow(data))
data_final_y_enc[data_final_y == 'MCI'] = 1

#################### FEATURE SELECTION ########################
# lasso
lambda <- 10^seq(10, -2, length = 100)
set.seed(1)
lasso.mod = train(
  Label ~., data = data, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda)
)
lasso.mod$bestTune$lambda

# Get the relevant features and their values
features = data.frame(as.matrix(coef(lasso.mod$finalModel, lasso.mod$bestTune$lambda)))
features$names = rownames(features)
features = features[features$s1 != 0,]
features = features[-2]
features_names = rownames(features)
features_names = features_names[-1]

sel_columns = colnames(data) %in% features_names
sel_columns[length(sel_columns)] = TRUE

##### Filtered data
data_filtered = data_train[, sel_columns]
data_filtered_test = data_test[, sel_columns]

data_filtered_final = data[, sel_columns]

##### set control for CV
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

################# LDA ################
set.seed(1)
lda.model <- train(Label ~ ., data = data_filtered, method = "lda", trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
print(lda.model) # ROC 0.9471

################ QDA #####################
set.seed(1)
qda.model <- train(Label ~ ., data = data_filtered, method = "qda", trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
print(qda.model) # ROC 0.5123

################ LOGISTIC REGRESSION #################
set.seed(1)
glm.model <- train(Label ~ ., data = data_filtered, method = "glm", trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
print(glm.model) # Too many predictors for this model, so we discard it even though ROC = 0.9531

################ NAIVE BAYES ####################
set.seed(1)
nb_grid <- expand.grid(usekernel = c(TRUE, FALSE), laplace = c(0, 0.5, 1), adjust = c(0.75, 1, 1.25, 1.5))
nb.model <- train(Label ~ ., data = data_filtered, method = "naive_bayes", usepoisson = TRUE, tuneGrid = nb_grid,  trControl = ctrl, preProcess=c("center","scale"), metric = 'ROC')
nb.model$finalModel$tuneValue
print(nb.model) # ROC  0.8538


############## LDA HAS THE HIGHEST AUC SO...
pred_prob_train <- predict(lda.model, data_filtered, type="prob")
pred_prob_val <- predict(lda.model, data_filtered_test, type="prob")

pred_class_train = predict(lda.model, data_filtered)
pred_class_test = predict(lda.model, data_filtered_test)

auc(data_train_y_enc,pred_prob_train[,1]) # 1
auc(data_test_y_enc,pred_prob_val[,1]) # 0.9375

mcc(pred_class_train,as.factor(data_train_y)) # 0.9712
mcc(pred_class_test,as.factor(data_test_y)) # 0.5221
# Therefore, we can conclude that our model did not overfit, as the mcc and the auc values are quite high and close to the train values

################################## FINAL MODEL ###################################
ctrl_final = trainControl(method = "none", classProbs = TRUE, summaryFunction = twoClassSummary)

# LDA MODEL WITH 38 VARIABLES ON WHOLE DATASET
lda.modelfinal = train(Label ~., data = data_filtered_final, method = "lda", preProcess=c("center","scale"), trControl = ctrl_final, metric = 'ROC')

# get metrics from training set
pred_prob_train = predict(lda.modelfinal, newdata = data_filtered_final, type = "prob")
pred_class_train = predict(lda.modelfinal, newdata = data_filtered_final)
auc(data_final_y_enc,pred_prob_train[,1]) # 0.9992
mcc(pred_class_train,as.factor(data_final_y)) # 0.9884

# do prediction on test_set
data_test = read.csv('MCICTLtest.csv')
data_id = data_test[,1]
data_test = data_test[,-1]

pred_prob_test = predict(lda.modelfinal, newdata = data_test, type = "prob")
pred_class_test = predict(lda.modelfinal, newdata = data_test)

ADCTLresults = data.frame(Id = data_id, Labels = pred_class_test, Probs = pred_prob_test[,1])
save(ADCTLresults, file = '0068100_Sam_MCICTLres.Rdata')

#get column indexes
Columns_index = which(names(data)%in%features_names) 
save(Columns_index, file = '0068100_Sam_MCICTLfeat.Rdata')


ROC_models = c(0.9471, 0.5123, 0.9531, 0.8538)
plot(ROC_models, type="b", xlab = "Model", ylab="AUC", las = 2, xaxt = "n", main="10-fold CV AUC for each model")
grid()
axis(1, at=1:4, labels=c("LDA", "QDA", "Log Reg", "Naive Bayes"))

