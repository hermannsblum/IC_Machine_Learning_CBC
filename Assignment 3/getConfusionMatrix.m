function [ confMatrix ] = getConfusionMatrix( labelsTest, predictions, numClasses )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
confMatrix = zeros(numClasses);
for j=1:length(predictions)
    confMatrix(labelsTest(j),predictions(j)) = confMatrix(labelsTest(j),predictions(j))+1;
end

end

