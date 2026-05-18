This is a fork of NanoChat focused on classifying grammatically correct sentences. It keeps the same core but changes the structure of the training and testing datasets. It also adds preference tuning and simplifies the training script by removing features I do not use.
datasets used: 
data/eval-input.tsv = leipzig corpus
data/devel.tsv = in this dataset, there are two sentences per line. one sentence is grammatically correct and the other one is incorrect, separated by a tab.