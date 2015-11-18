function [ parameters, accuracies ] = crossValidate( algorithm, attributes, labels )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Get the indices of the 10 folds as a cell array of 10 indices arrays
foldsIndices = getFoldsPartitioning(labels,10,true);
[attributesNN,labelsNN] = ANNdata(attributes,labels);

[parameters, numParams] = getParameters(algorithm);
accuracies = zeros(numParams);
for i=1:10
    trainingSetIndices = getTrainingSetIndexed(foldsIndices,i);
    validationSetIndices = foldsIndices{i};
    
    accuraciesPerFold = validateNeuralNetwork(algorithm,parameters,attributesNN,labelsNN,trainingSetIndices,validationSetIndices);
    
    accuracies = accuracies+accuraciesPerFold;
end
% Average the accuracies
accuracies = accuracies./10;



end

