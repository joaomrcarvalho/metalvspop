#setwd("whatever your path is")
library(tm)
library(wordcloud)

metal.train <- Corpus(DirSource("metalTraining", encoding = "UTF-8"))
pop.train <- Corpus(DirSource("popTraining", encoding = "UTF-8"))
metal.test <- Corpus(DirSource("metalTest", encoding = "UTF-8"))
pop.test <- Corpus(DirSource("popTest", encoding = "UTF-8"))

docsMetal <- c(metal.train, metal.test)
docsPop <- c(pop.train, pop.test)

#variaveis
l1 <- length(metal.train)
l2 <- length(pop.train)
l3 <- length(metal.test)
l4 <- length(pop.test)
minimumWordLenght <- 3
maximumLenght <- 20
minimumDocumentFrequency <- 1

docsMetal <- tm_map(docsMetal, PlainTextDocument)
docsMetal <- tm_map(docsMetal, removeWords, stopwords())
docsMetal <- tm_map(docsMetal, stripWhitespace)
docsMetal <- tm_map(docsMetal, content_transformer(tolower))
docsMetal <- tm_map(docsMetal, removePunctuation)
docsMetal <- tm_map(docsMetal, removeNumbers)

docsPop <- tm_map(docsPop, PlainTextDocument)
docsPop <- tm_map(docsPop, removeWords, stopwords())
docsPop <- tm_map(docsPop, stripWhitespace)
docsPop <- tm_map(docsPop, content_transformer(tolower))
docsPop <- tm_map(docsPop, removePunctuation)
docsPop <- tm_map(docsPop, removeNumbers)

docsTermMatrixMetal <- DocumentTermMatrix(docsMetal, control=list(wordLengths=c(minimumWordLenght, maximumLenght), bounds = list(global = c(minimumDocumentFrequency,Inf))))
docsTermMatrixPop <- DocumentTermMatrix(docsPop, control=list(wordLengths=c(minimumWordLenght, maximumLenght), bounds = list(global = c(minimumDocumentFrequency,Inf))))

dtmMetal <- as.matrix(docsTermMatrixMetal)
frequencyMetal <- colSums(dtmMetal)
frequencyMetal <- sort(frequencyMetal, decreasing = TRUE)
wordsMetal <- names(frequencyMetal)
wordcloud(wordsMetal[1:100], frequencyMetal[1:100])

dtmPop <- as.matrix(docsTermMatrixPop)
frequencyPop <- colSums(dtmPop)
frequencyPop <- sort(frequencyPop, decreasing = TRUE)
wordsPop <- names(frequencyPop)
wordcloud(wordsPop[1:100], frequencyPop[1:100])

interseccao <- intersect(wordsPop[1:100], wordsMetal[1:100])
