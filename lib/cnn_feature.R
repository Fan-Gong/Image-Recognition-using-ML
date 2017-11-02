

extract_feature <- function(img_dir, width =64, height = 64,ratio = 0.9, is_test =T) {
  library(caTools)
  library(EBImage)
  library(pbapply)
  
    img_size = width*height
    images_names  = list.files(path=img_dir, pattern = ".*.jpg")
    
    print(paste("Start processing", length(images_names), "images"))
    feature_list <- pblapply(images_names, function(imgname) {
      
      img <- readImage(file.path(img_dir, imgname))
      img_resized <- resize(img, w = width, h = height)
      grayimg <- channel(img_resized, "gray")
      img_matrix = grayimg@.Data
      img_vector = as.vector(t(img_matrix))
      return(img_vector)
    })
    feature_matrix <- do.call(rbind, feature_list)  ## bind the list of vector into matrix
    feature_matrix <- as.data.frame(feature_matrix)
    names(feature_matrix) <- paste0("pixel", c(1:img_size))
    
    if (is_test ==T){
      if(export){
        write.table(cnn_feature,file=paste0("../output/cnn_feature_",cell,orientation,"_", data_name, "_", set_name, ".csv"),
                    row.names = F,col.names = F,sep=",")
      }
      
      return(feature_matrix)
    }
    
  
    label = read.csv("/Users/bxin66/Dropbox/Columbia/Class3/5243/project3/train_set/label_train.csv", header = T)
    feature_matrix = cbind(label[,2],feature_matrix)
    
    feature_matrix = cbind(sample.split(label[,2],SplitRatio=ratio),feature_matrix)
    colnames(feature_matrix)[1:2] = c("is_train","label")
    
    df_second = data.frame(matrix(NA, nrow = ratio*dim(feature_matrix)[1], ncol = (img_size+2)))
    df_second[,1:2] = feature_matrix[feature_matrix$is_train==T,1:2]
    
    for (i in 1:sum(feature_matrix$is_train==T)){
      df_second[i,3:4098] = unlist(t(flop(t(matrix(feature_matrix[feature_matrix$is_train==T,][i,3:4098],nrow = 64,ncol = 64)))))
    }
    names(df_second)[3:4098] = paste0("pixel", c(1:4096))
    names(df_second)[1:2] = c("is_train","label")
    
    
    if(export){
      write.table(cnn_feature,file=paste0("../output/cnn_feature_",cell,orientation,"_", data_name, "_", set_name, ".csv"),
                  row.names = F,col.names = F,sep=",")
    }
    
  return(rbind(feature_matrix,df_second))
  
}


