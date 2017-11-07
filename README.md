# ambiguity-resolving-ANN
Contains MATLAB code for a Multi Layer Perceptron (MLP) that can be used to resolve ambiguity (due to multiple labels for a training point) and to assign a non-ambiguous label to a test point. multiTargetANN is a MATLAB code for an ambiguity resolving MLP. The input is a dataset where the data points are in the rows, and the corresponding class labels are in a cell array. multiTargetAll1, multiTargetRand2 and multiTargetMin3 are all alternative approaches to solve ambiguity. The supporting function required by all of the scripts is uniqueCell.
Further details can be found in the corresponding research article:
Shounak Datta, Sankha Subhra Mullick, Swagatam Das, 2017, Generalized mean based back-propagation of errors for ambiguity resolution, Pattern Recognition Letters, 94, pp:22-29.

