function [trS, labelsTrS] = getTrainingSet(folds,labelsFolds,i)
% Returns the training set for the i-th iteration of cross-validation

trS = [];
labelsTrS = [];
for j=1:length(folds)
    if(j~=i)
        trS = [trS; folds{j}];
        labelsTrS = [labelsTrS; labelsFolds{i}];
    end
end