#setwd("whatever your path is")
library(tm)
library(wordcloud)
library(rpart)
library(nnet)
library(RWeka)
library(adabag)
library(cvTools)
library(class)

tempoInicial <- as.double(proc.time()[3])

cat("----- LOADING DATA ----- \n")
metal.train <- Corpus(DirSource("metalTraining", encoding = "UTF-8"))
pop.train <- Corpus(DirSource("popTraining", encoding = "UTF-8"))
metal.test <- Corpus(DirSource("metalTest", encoding = "UTF-8"))
pop.test <- Corpus(DirSource("popTest", encoding = "UTF-8"))
#metal.test <- Corpus(DirSource("metalTestAlternativo", encoding = "UTF-8"))
#pop.test <- Corpus(DirSource("popTestAlternativo", encoding = "UTF-8"))
#metal.test <- Corpus(DirSource("metalTestAlternativo2", encoding = "UTF-8"))
#pop.test <- Corpus(DirSource("popTestAlternativo2", encoding = "UTF-8"))

tempoLoadingData <- as.double(proc.time()[3])

docs <- c(metal.train, pop.train, metal.test, pop.test)


#variaveis
l1 <- length(metal.train)
l2 <- length(pop.train)
l3 <- length(metal.test)
l4 <- length(pop.test)
minimumWordLenght <- 3
maximumLenght <- 20
minimumDocumentFrequency <- as.integer((l1+l2+l3+l4)/50)

#preprocessing
docs <- tm_map(docs, PlainTextDocument)
docs <- tm_map(docs, removeWords, stopwords())
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, stemDocument)

docsTermMatrixGlobal <- DocumentTermMatrix(docs, control=list(wordLengths=c(minimumWordLenght, maximumLenght), bounds = list(global = c(minimumDocumentFrequency,Inf))))

docsTermMatrixGlobal <- as.data.frame(inspect(docsTermMatrixGlobal))

classVector <- c(rep("metal",l1),rep("pop",l2),rep("metal",l3),rep("pop",l4))

docsTermMatrixGlobal <- cbind(docsTermMatrixGlobal, classVector)


j <- 0
columnsToKeep <- c()
for(i in 1:(length(docsTermMatrixGlobal)-1)) {
  if(sum(docsTermMatrixGlobal[i]) > (0.15*(l1+l2+l3+l4))) {
    j <- j+1
    columnsToKeep <- c(columnsToKeep, i)
  }
}

columnsToKeep <- c(columnsToKeep, length(docsTermMatrixGlobal))

docsTermMatrixGlobal <- docsTermMatrixGlobal[,columnsToKeep]

last.col <- length(docsTermMatrixGlobal)
docsTermMatrix.tr <- docsTermMatrixGlobal[1:(l1+l2), 1:last.col]
docsTermMatrix.test <- docsTermMatrixGlobal[(l1+l2+1):(l1+l2+l3+l4), 1:(last.col-1)]

info.terms <- colnames(docsTermMatrixGlobal)[1:length(docsTermMatrixGlobal-1)]

rename.terms.in.list <- function(list) {
  for (i in 1:length(list)) {
    list[i]<- paste(list[i],".t", sep="")
  } 
  return(list) 
}

info.terms<-rename.terms.in.list(info.terms)

rename.terms.in.dtm <- function(dtm) {
  for (i in 1:length(dtm)) {
    #cat("replaced", names(dtm)[i], "at", i, "with", paste(names(dtm)[i],".t", sep=""), "\n")
    names(dtm)[i] <- paste(names(dtm)[i],".t", sep="")
  }
  return(dtm)
}

docsTermMatrix.tr<-rename.terms.in.dtm(docsTermMatrix.tr)
docsTermMatrix.test<-rename.terms.in.dtm(docsTermMatrix.test)

names.tr <-paste(info.terms, collapse='+')

clas.formula <- as.formula( paste('classVector.t', names.tr, sep='~') ) 

replaceIntegersPerClasses <- function(preds) {
  for(i in 1:length(preds)) {
    if(preds[i] == "1") preds[i] <- "metal"
    
    if(preds[i] == "2") preds[i] <- "pop"  
  }
  
  return(preds)
}


tempoPreprocessamento <- as.double(proc.time()[3])

cat("----- TRAINING CLASSIFIERS ----- \n")

#treinar classificadores
neuralNetClassifier1 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 2, maxit = 300)
neuralNetClassifier2 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 2, maxit = 800)
neuralNetClassifier3 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 3, maxit = 300)
neuralNetClassifier4 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 3, maxit = 800)
neuralNetClassifier5 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 4, maxit = 300)
neuralNetClassifier6 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 4, maxit = 800)
decisionTreeClassifier7 <- rpart(clas.formula, docsTermMatrix.tr)
NB<-make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
NBClassifier8<-NB(clas.formula, docsTermMatrix.tr)
neuralNetClassifier10 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 2, maxit = 400)
neuralNetClassifier11 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 2, maxit = 600)
neuralNetClassifier12 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 3, maxit = 400)
neuralNetClassifier13 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 3, maxit = 600)
neuralNetClassifier14 <- nnet(clas.formula, data=docsTermMatrix.tr, size = 4, maxit = 400)
decisionTreeClassifier15 <- rpart(clas.formula, docsTermMatrix.tr)
NB2<-make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
NBClassifier16<-NB2(clas.formula, docsTermMatrix.tr)


tempoTreinarClassificadores <- as.double(proc.time()[3])


#predictions
cat("----- PREDICTING ON TEST DATA ----- \n")
predictions1 <- predict(neuralNetClassifier1, docsTermMatrix.test, type="class")
predictions2 <- predict(neuralNetClassifier2, docsTermMatrix.test, type="class")
predictions3 <- predict(neuralNetClassifier3, docsTermMatrix.test, type="class")
predictions4 <- predict(neuralNetClassifier4, docsTermMatrix.test, type="class")
predictions5 <- predict(neuralNetClassifier5, docsTermMatrix.test, type="class")
predictions6 <- predict(neuralNetClassifier6, docsTermMatrix.test, type="class")
predictions7 <- predict(decisionTreeClassifier7, docsTermMatrix.test, type="class")
predictions8 <- predict(NBClassifier8, docsTermMatrix.test)
tempMatrix <- docsTermMatrix.tr
tempMatrix[tempMatrix$classVector.t<-NULL]
predictions9 <- knn(tempMatrix, docsTermMatrix.test, cl=factor(c(rep("metal",1890),rep("pop",1153))), k=10)
predictions10 <- predict(neuralNetClassifier10, docsTermMatrix.test, type="class")
predictions11 <- predict(neuralNetClassifier11, docsTermMatrix.test, type="class")
predictions12 <- predict(neuralNetClassifier12, docsTermMatrix.test, type="class")
predictions13 <- predict(neuralNetClassifier13, docsTermMatrix.test, type="class")
predictions14 <- predict(neuralNetClassifier14, docsTermMatrix.test, type="class")
predictions15 <- predict(decisionTreeClassifier15, docsTermMatrix.test, type="class")
predictions16 <- predict(NBClassifier16, docsTermMatrix.test)
predictions17 <- predictions9


cat("----- COMBINING THE PREDICTION MATRICES ----- \n")
generalPredictionsMatrix <- rbind(predictions1,predictions2,predictions3,predictions4,predictions5,predictions6,
                                  predictions7,predictions8,predictions9,
                                  predictions10,predictions11,predictions12,predictions13,predictions14,
                                  predictions15,predictions16,predictions17)



generalPredictionsMatrix <- replaceIntegersPerClasses(generalPredictionsMatrix)

finalPredictionsMatrix <- c()

 for(i in 1:ncol(generalPredictionsMatrix)) {
   numMetal <- 0
   
   for(j in 1:nrow(generalPredictionsMatrix)) {
     if(generalPredictionsMatrix[j,i] == "metal") numMetal <- numMetal + 1
   }
   
   if(numMetal > nrow(generalPredictionsMatrix)/2) {
     #metal
     finalPredictionsMatrix <- c(finalPredictionsMatrix, "metal")
   }else {
     #pop
     finalPredictionsMatrix <- c(finalPredictionsMatrix, "pop")
   }
   
 } 

tempoPredictions <- as.double(proc.time()[3])

calcPerformance <- function(predictions) {
  # #positivo -> METAL
  # #negativo -> POP
  # #metal classificado como pop -> False Negative
  # #pop classificado como metal -> False Positive
  # #metal classificado como metal -> True positive
  # #pop classificado como pop -> True Negative
  FN <- 0
  FP <- 0
  TP <- 0
  TN <- 0
  
  #metal
  for(i in 1:l3)
  {
    if(predictions[i] == "metal") {
      TP <- TP + 1
    }
    
    if(predictions[i] == "pop") {
      FN <- FN + 1
    }
  }
  
  #pop
  for(i in l3+1:l4)
  {
    if(predictions[i] == "metal") {
      FP <- FP + 1
    }
    
    if(predictions[i] == "pop") {
      TN <- TN + 1
    }
  }
  cat("------- MEASURES -------- \n")
  accuracyRate <- (TP+TN)/(TP+FN+FP+TN)
  cat("accuracy rate = ")
  cat(accuracyRate)
  cat("\n")
  
  recall <- TP / (TP+FN)
  cat("recall = ")
  cat(recall)
  cat("\n")
  
  precision <- TP / (TP+FP)
  cat("precision = ")
  cat(precision)
  cat("\n")
  
  f1 <- 2 * precision * recall / (precision+recall)
  cat("f1 = ")
  cat(f1)
  cat("\n")
  
  cat("------- CONFUSION MATRIX --------\n")
  cat("         |  metal   |   pop  \n")
  cat("-----------------------------\n")
  cat("metal    |  ")
  cat(TP)
  cat("     |   ")
  cat(FN)
  cat("   \n")
  cat("pop      |  ")
  cat(FP)
  cat("       |   ")
  cat(TN)
  cat("   \n")
}

calcPerformance(finalPredictionsMatrix)

cat("Number of variables (words) used: ")
cat(ncol(docsTermMatrixGlobal))
cat("\n")


cat("------- TIME (SECONDS) -------- \n")
cat("Loading data: ")
t1 <- tempoLoadingData-tempoInicial
cat(t1)
cat("\n")


cat("Preprocessing data: ")
t2 <- tempoPreprocessamento-tempoInicial-t1
cat(t2)
cat("\n")

cat("Training classifiers: ")
t3 <- tempoTreinarClassificadores-tempoInicial-t1-t2
cat(t3)
cat("\n")

cat("Predicting on test set data: ")
t4 <- tempoPredictions-tempoInicial-t1-t2-t3
cat(t4)
cat("\n")

############################################
# 
# dtmMetal <- as.matrix(docsTermMatrixMetalTrain)
# frequencyMetal <- colSums(dtmMetal)
# frequencyMetal <- sort(frequencyMetal, decreasing = TRUE)
# wordsMetal <- names(frequencyMetal)
# wordcloud(wordsMetal[1:100], frequencyMetal[1:100])
# 
# 
# dtmPop <- as.matrix(docsTermMatrixPopTrain)
# frequencyPop <- colSums(dtmPop)
# frequencyPop <- sort(frequencyPop, decreasing = TRUE)
# wordsPop <- names(frequencyPop)
# wordcloud(wordsPop[1:100], frequencyPop[1:100])
# 
# interseccao <- intersect(wordsPop[1:100], wordsMetal[1:100])
#
#######################################
# docsTermMatrixMetalTrain <- DocumentTermMatrix(metal.train, control=list(wordLengths=c(minimumWordLenght, maximumLenght), bounds = list(global = c(minimumDocumentFrequency,Inf))))
# docsTermMatrixMetalTest <- DocumentTermMatrix(metal.test, control=list(wordLengths=c(minimumWordLenght, maximumLenght), bounds = list(global = c(minimumDocumentFrequency,Inf))))
# docsTermMatrixPopTrain <- DocumentTermMatrix(pop.train, control=list(wordLengths=c(minimumWordLenght, maximumLenght), bounds = list(global = c(minimumDocumentFrequency,Inf))))
# docsTermMatrixPopTest <- DocumentTermMatrix(pop.test, control=list(wordLengths=c(minimumWordLenght, maximumLenght), bounds = list(global = c(minimumDocumentFrequency,Inf))))

#freqterms100MetalTrain <- findFreqTerms(docsTermMatrixMetalTrain, 200)
#freqterms100PopTrain <- findFreqTerms(docsTermMatrixPopTrain, 200)
#freqterms100Geral <- findFreqTerms( docsTermMatrixImproved, 100)
#temporario..para testes
#currentTrainMatrix <- docsTermMatrixMetalTrain
