function [ testSetIndices, trainingSetIndices ] = holdOutTestSet( labels, proportion, stratified )
%Separates a test set from the dataset
%Proportion is a number between 0 and 1 indicating the fraction of examples
%to put in the held out test set

testSetIndices = [];
if(stratified)
    indicesPerClass = stratifySampleIndexed(labels);
    for i=1:length(indicesPerClass)
        testSetIndices = [testSetIndices; indicesPerClass{i}(1:1:floor(proportion*length(indicesPerClass{i})))'];
    end
    testSetIndices = sort(testSetIndices);
else
    testSetIndices = 1:floor(proportion*length(labels));
end
trainingSetIndices = setdiff(1:length(labels),testSetIndices);

end

