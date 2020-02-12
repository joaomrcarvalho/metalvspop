# Metal Vs Pop Lyrics - Machine Learning using R
R Project to distinguish between Metal songs and Pop songs -- only based on their lyrics.

The directories contain lyrics downloaded from AzLyrics website using AzLyrics Downloader (https://github.com/joaomrcarvalho/AzLyrics-Downloader).

- Main database:
	- "metalTraining" -> lyrics from metal bands used for training the classifiers
	- "popTraining" -> lyrics from metal bands used for training the classifiers
	- "metalTest" -> lyrics from metal bands used to test the classifiers obtained
	- "popTest" -> lyrics from metal bands used to test the classifiers obtained

Validation data:
	- "metalTestAlternativo" -> lyrics from metal bands used to test the classifiers obtained
	- "popTestAlternativo" -> lyrics from metal bands used to test the classifiers obtained
	- "metalTestAlternativo2" -> lyrics from metal bands used to test the classifiers obtained
	- "popTestAlternativo2" -> lyrics from metal bands used to test the classifiers obtained

- "main.R": R script used to implement the project
- "wordclouds.R": R script used to plot wordclouds from both data sets
