#install.packages("OpenImageR")
library(OpenImageR)

train_data<-list()
img_dir<-"../data/training_set/images"


for (i in 1:3000){
  n<-nchar(as.character(i))
  path<-paste0(img_dir,"/img_",paste(rep(0,4-n),collapse=""),i,".jpg")
  train_data[[i]]<-readImage(path)
}


hog <- vector()

for (i in 1:3000){
  hog <- rbind(hog,HOG(train_data[[i]]))
}

write.csv(hog,file="../output/hog.csv")


