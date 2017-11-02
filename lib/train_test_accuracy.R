###############################################################################
#            Calculating the Training and Testing Accuracy                    #
#                                                                             #
###############################################################################

train_test_accuracy = function(feature) {
  
  ### feature: feature extraction method we use
  
  source("../lib/train.R")
  source("../lib/test.R")
  source("../lib/train_test_split.R")
  library(caret)
  library(gbm)
  library(e1071)
  library(randomForest)
  library(nnet)
  
  df = train_test_split(feature_name = feature) 
  df_train = df$df_train
  df_test = df$df_test
  
  
  ### lr
  lr = train_lr(df_train)
  
  train_lr_accuracy = mean(test(lr,df_train) == df_train$Label)
  test_lr_accuracy = mean(test(lr,df_test) == df_test$Label)
  lr_time = lr$time
  
  
  ### GBM
  
  gbm = train_gbm(df_train)
  
  train_gbm_accuracy = mean(test_gbm(gbm,df_train) == df_train$Label)
  test_gbm_accuracy = mean(test_gbm(gbm,df_test) == df_test$Label)
  gbm_time = gbm$time
  
  
  ### Random Forest
  rf = train_rf(df_train)
  
  train_rf_accuracy = mean(test(rf,df_train) == df_train$Label)
  test_rf_accuracy = mean(test(rf,df_test) == df_test$Label)
  rf_time = rf$time
  
  ### SVM
  svm = train_svm(df_train)
  
  train_svm_accuracy = mean(test(svm,df_train) == df_train$Label)
  test_svm_accuracy = mean(test(svm,df_test) == df_test$Label)
  svm_time = svm$time
  
  
  ### Result Table
  df_result = data.frame(Model = c('Random Forest', 'Logistic Regression', 
                                   'SVM', 'GBM'), 
                         Train_accuracy = c(train_rf_accuracy, train_lr_accuracy, 
                                            train_svm_accuracy, train_gbm_accuracy), 
                         Test_accuracy = c(test_rf_accuracy, test_lr_accuracy, 
                                           test_svm_accuracy, test_gbm_accuracy), 
                         Running_Time = c(rf_time, lr_time, svm_time, gbm_time))
  
  
  return(df_result)
  
}


