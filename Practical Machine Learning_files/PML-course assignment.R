## 0. Initialize and get started....

set.seed(123)
library(caret)
library(randomForest)
library(ggplot2)
##library(Hmisc)
library(foreach)
library(doParallel)


## 1. Download the data sets and review them

trainRawDat <- read.csv("pml-training.csv") 
testRawDat <- read.csv("pml-testing.csv")

print("number of observations/variables:") 
print(dim(trainRawDat))
##cat ("Press [enter] to continue")
##line <- readline()

print("Printing summary of Raw Data:") 
print(summary(trainRawDat))
##cat ("Press [enter] to continue")
##line <- readline()

## 2. Clean and prepare the observations 
## take out variables with a high number of "NA's"
## check percentage of NA's for each variable
numNA=integer()
pctNA=numeric()

for(n in 1: ncol(trainRawDat)){
	numNA[n]<-sum(is.na(trainRawDat[,n]))
	pctNA[n]<-numNA[n]/nrow(trainRawDat)
}

## print(pctNA)

## eliminate the variables with >97% NA's
goodVAR<- pctNA<0.97
cleanDat<-trainRawDat[,goodVAR]
test4real<-testRawDat[,goodVAR]

## after review realized that we can also delete first 6 variables
cleanDat<-cleanDat[,c(7:ncol(cleanDat))]
test4real<-test4real[,c(7:ncol(test4real))]

## review raw test data for NA's and also exclude if >97% NA's
numNA2=integer()
pctNA2=numeric()

for(n in 1: ncol(test4real)){
	numNA2[n]<-sum(is.na(test4real[,n]))
	pctNA2[n]<-numNA2[n]/nrow(test4real)
}
goodVAR2<- pctNA2<0.97
cleanDat<-cleanDat[,goodVAR2]
test4real<-test4real[,goodVAR2]


## Convert Classification-feature to factor, all others as numeric
cleanDat$classe <-as.factor(cleanDat$classe)
max=ncol(cleanDat)-1
cleanDat[,1:max]<- sapply(cleanDat[,1:max], as.numeric)
max=ncol(test4real)-1
test4real[,1:max]<- sapply(test4real[,1:max], as.numeric)

## 3. Split the training data set
## testing is NOTthe final test: its the cross validation test set
inTrain <- createDataPartition(y=cleanDat$classe, p=0.85, list=FALSE)
training <- cleanDat[inTrain,]
testing <- cleanDat[-inTrain,]
training$classe <- as.factor(training$classe)
testing$classe <- as.factor(testing$classe)

## 4. Plotting and reviewing the data - just playing!
##featurePlot(x=cleanDat[,c(7,8,9,10)],y=training$classe, plot="pairs")
##qplot(accel_forearm_z, accel_forearm_x,color=classe, data=training)
##qplot(accel_forearm_z, color=classe, data=training, geam="density")

## 5. Building the model
set.seed(3689)
registerDoParallel()
fitControl <- trainControl(method = "repeatedcv", number = 2, repeats = 2)
modelFitRF <- train(classe ~ ., data = training, method = "rf", preProcess="pca", trControl = fitControl)
modelFitGBM<- train(classe ~ ., data = training, method = "gbm", preProcess="pca", trControl = fitControl)


## 6. Testing the model
## predict using RF
print(modelFitRF)
predTrain <- predict(modelFitRF,training)
table(predTrain, training$classe)
predTest <- predict(modelFitRF,testing)
table(predTest, testing$classe)
conMatRF<- confusionMatrix(testing$classe, predict(modelFitRF, testing))
print(conMatRF)


## predict using GBM
print(modelFitGBM)
predTrain <- predict(modelFitGBM,training)
table(predTrain, training$classe)
predTest <- predict(modelFitGBM,testing)
table(predTest, testing$classe)
conMatGBM<- confusionMatrix(testing$classe, predict(modelFitGBM, testing))
print(conMatGBM)

## RF predicts the better/more accurate results (98+%!) - use RF

## 7. run on test data (4 real!) and product output files

## resultsGBM <- as.character(predict(modelFit,test4real))
resultsRF<- as.character(predict(modelFitRF,test4real))

pml_write_files = function(x){
	n = length(x)
	for(i in 1:n){
	filename = paste0("problem_id_",i,".txt")
	write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
	}
}
pml_write_files(resultsRF)

