##########This is the file for finding the parameters of hog, it will take a long time to run#######



library(caret)

label = read.csv('../data/training_set/label_train.csv', header = T)

or<- c(6,8,10)
cell<-c(5,6,7,8)

library(OpenImageR)

 train_data<-list()
 img_dir<-"../data/training_set/images"
 
 
 for (i in 1:3000){
   n<-nchar(as.character(i))
   path<-paste0(img_dir,"/img_",paste(rep(0,4-n),collapse=""),i,".jpg")
   train_data[[i]]<-readImage(path)
 }

accuracy<-matrix(NA, nrow = length(cell), ncol =length(or))
colnames(accuracy) <- paste("orientation", or)
rownames(accuracy) <- paste("cells", cell)

  for(j in 1:length(cell)){
    for(h in 1: length(or)){
      
      hog <- vector()
      for (i in 1:3000){
        hog <- rbind(hog,HOG(train_data[[i]],cells=cell[j],orientations = or[h]))
      }
      
      #write.csv(hog,file=paste0("hog",cell[j],or[h],".csv"))
      #write.table(hog,file=paste0("hog",cell[j],or[h],".csv"),row.names = F,col.names = F,sep=",")
      
      df=hog
      set.seed(42)
      df_complete = cbind(label, df) # Combine the dataframe and label
      df_complete = df_complete[, -c(1)] # Delete the Useless Column 
      colnames(df_complete)[1] = c('Label') # Rename the Column
      df_complete$Label = as.factor(df_complete$Label) # Make the Label as factor
      dpart = createDataPartition(df_complete$Label, p = 0.2, list = F) #Balanced Partition
      df_test = df_complete[dpart, ] # test set
      df_train = df_complete[-dpart, ] # training set

      lr.fit = multinom(Label~., data = df_train, MaxNWts=16000)
      accuracy[j,h] <-1- mean(predict(lr.fit, df_test) != df_test$Label)

    }
    
  }

save(accuracy,file="../output/accuracy.Rdata")

