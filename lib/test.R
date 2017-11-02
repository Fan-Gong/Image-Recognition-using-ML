######################################################
### Fit the classification model with testing data ###
######################################################

### Author: Fan Gong
### Project 3
### ADS Fall 2016


##### Testing Function For GBM Model #####

test_gbm = function(fit_train, dat_test){
  library("gbm")
  pred = predict(fit_train$fit, newdata=dat_test,n.trees = 400, type="response")
  gbm_prob = predict(fit_train$fit, n.trees = 400, newdat =  dat_test, type="response")
  gbm_class = apply(gbm_prob, 1, which.max) - 1
  return(gbm_class)
}


##### Testing Function For CNN Model #####

test_cnn = function(model_path,iteration = iter, dat_test_cnn){
  test_x = t(dat_test_cnn[dat_test_cnn$is_train==F,3:4098])
  test_y = dat_test_cnn[dat_test_cnn$is_train==F,2]
  
  model_cnn = mx.model.load(model_path,iteration = iter)
  predicted <- predict(model_cnn,test_array)
  return(max.col(t(predicted)) - 1)
  
}


##### Test Function For All Other Classification Model #####
test = function(fit_train, dat_test){
  pred = predict(fit_train$fit, newdata = dat_test)
  return(pred)
}










