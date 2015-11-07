function [trS, labelsTrS] = getTrainingSet(folds,labelsFolds,i)
% Returns the training set for the i-th iteration of cross-validation.
% It is just a concatenation of all the folds except the i-th

trS = [];
labelsTrS = [];
for j=1:length(folds)
    if(j~=i)
        trS = [trS; folds{j}];
        labelsTrS = [labelsTrS; labelsFolds{j}];
    end
end