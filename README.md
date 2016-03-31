# ambiguity-resolving-ANN
Contains MATLAB code for a Multi Layer Perceptron that can be used to resolve ambiguity (generated from multiple labels of a training point) and successfully classify a test point.
multiTargetANN is a MATLAB code for an ambigutiy resolving multi layer perceptron. The input is a dataset where the data points are in the rows, and the class labels is respresented by a cell array. 
multiTargetAll1, multiTargetRand2 and multiTargetMin3 are all alternative approaches to solve ambigutiy.
The supporting function required by all of the scripts is uniqueCell
