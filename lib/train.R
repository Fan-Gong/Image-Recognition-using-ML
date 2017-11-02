#########################################################
### Train a classification model with training images ###
#########################################################

### Author: Fan Gong
### Project 3
### ADS Fall 2016


##### GBM #####

train_gbm = function(dat_train){
  
  ###  dat_train: processed features from images also contains label
  
  library(gbm)
  
  start_time_gbm = Sys.time() # Model Start Time
  
  gbm.fit = gbm(Label~., 
                data = dat_train,
                n.trees = 400,
                distribution = "multinomial",
                interaction.depth = 3, 
                shrinkage = 0.2,
                n.minobsinnode = 30,
                verbose=FALSE)
  end_time_gbm = Sys.time() # Model End time
  gbm_time = end_time_gbm - start_time_gbm #Total Running Time
  
  
  return(list(fit = gbm.fit, time = gbm_time))
}



##### SVM #####

train_svm = function(dat_train){

  ###  dat_train: processed features from images also contains label

  library(e1071)
  
  fitControl = trainControl(method = 'cv', number = 2)
  
  svmGrid = expand.grid(C = c(0.5,1,2))
  
  start_time_svm = Sys.time() # Model Start Time
  svm.fit = train(Label~., 
                  data = dat_train,
                  method = "svmLinear",
                  preProc = c('center', 'scale'),
                  tuneGrid = svmGrid,
                  trControl = fitControl)
  end_time_svm = Sys.time() # Model End time
  
  svm_time = end_time_svm - start_time_svm #Total Running Time
  
  
  return(list(fit = svm.fit, time = svm_time))
  
}


##### Logistic Regression #####

train_lr = function(dat_train){
  
  ###  dat_train: processed features from images also contains label
  
  library(nnet)
  
  start_time_lr = Sys.time() # Model Start Time
  lr.fit = multinom(Label~., 
                    data = dat_train, 
                    MaxNWts=16000)
  end_time_lr = Sys.time() # Model End time
  end_time_lr - start_time_lr
  
  lr_time = end_time_lr - start_time_lr #Total Running Time
  
  
  return(list(fit = lr.fit, time = lr_time))
}


##### Random Forest #####

train_rf = function(dat_train){
  
  ###  dat_train: processed features from images also contains label
  
  fitControl = trainControl(method = 'cv', number = 2)
  
  rfGrid = expand.grid(mtry = floor(sqrt(ncol(dat_train)) * 0.95) : floor(sqrt(ncol(dat_train) * 1.05)))
  
  start_time_rf = Sys.time() # Model Start Time 
  rf.fit = train(Label~., 
                 data = dat_train,
                 method = "rf", 
                 trControl = fitControl,
                 ntree = 500, #number of trees to grow
                 tuneGrid = rfGrid) # Parameter Tuning
  end_time_rf = Sys.time() # Model End time
  end_time_rf - start_time_rf
  
  rf_time = end_time_rf - start_time_rf #Total Running Time
  
  return(list(fit = rf.fit, time = rf_time))
}


##### Neural Networks #####

train_cnn = function (dat_train_cnn){
  start_time_cnn = Sys.time()
  
  train_x = t(df_all[df_all$is_train==T,3:4098])
  train_y = c(df_all[df_all$is_train==T,2])
  
  train_array = train_x
  dim(train_array) = c(64, 64, 1, ncol(train_x))
  
  test_data = df_all[df_all$is_train==F,2:4098]
  test_x = t(df_all[df_all$is_train==F,3:4098])
  test_y = df_all[df_all$is_train==F,2]
  
  test_array = test_x
  dim(test_array) = c(64, 64, 1, ncol(test_x))
  
  
  data <- mx.symbol.Variable('data')
  # 1st convolutional layer
  conv_1 = mx.symbol.Convolution(data = data, kernel = c(4, 4), num_filter = 20)
  act_1 = mx.symbol.Activation(data = conv_1, act_type = "tanh")
  pool_1 = mx.symbol.Pooling(data = act_1, pool_type = "max", kernel = c(7, 7), stride = c(2, 2))
  dropout_1 = mx.symbol.Dropout(data = pool_1, p=0.3)
  # 2nd convolutional layer
  conv_2 = mx.symbol.Convolution(data = dropout_1, kernel = c(4, 4), num_filter = 50)
  act_2 = mx.symbol.Activation(data = conv_2, act_type = "tanh")
  pool_2 = mx.symbol.Pooling(data=act_2, pool_type = "max", kernel = c(7, 7), stride = c(2, 2))
  dropout_2 = mx.symbol.Dropout(data = pool_2, p=0.3)
  
  
  # 1st fully connected layer
  flatten = mx.symbol.Flatten(data = dropout_2)
  
  fc_1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
  nb_1 = mx.symbol.BatchNorm(data = fc_1)
  act_3 = mx.symbol.Activation(data = nb_1, act_type = "tanh")
  dropout_3 = mx.symbol.Dropout(data = act_3, p=0.3)
  
  # 2nd fully connected layer
  fc_2 = mx.symbol.FullyConnected(data = dropout_3, num_hidden = 3)
  nb_2 = mx.symbol.BatchNorm(data = fc_2)
  
  # Output. Softmax output since we'd like to get some probabilities.
  NN_model <- mx.symbol.SoftmaxOutput(data = nb_2)
  
  mx.set.seed(100)
  devices <- mx.cpu()
  # Train the model
  model <- mx.model.FeedForward.create(NN_model,
                                       X = train_array,
                                       y = train_y,
                                       ctx = devices,
                                       num.round = 45,
                                       array.batch.size = 64,
                                       learning.rate = 0.001,
                                       eval.metric = mx.metric.accuracy,
                                       initializer = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
                                       optimizer = "adam",
                                       #epoch.end.callback = mx.callback.log.train.metric(100),
                                       epoch.end.callback = mx.callback.save.checkpoint("train_cnn"))
  
  
  end_time_cnn = Sys.time() # Model End time
  cnn_time = end_time_cnn - start_time_cnn #Total Running Time
  return(cnn_time)
}







