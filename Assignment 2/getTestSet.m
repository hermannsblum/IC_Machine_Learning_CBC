function [ts, labelsTs] = getTestSet(folds,labelsFolds,i)
% Returns the test set for the i-th iteration of the cross-validation

ts = folds{i};
labelsTs = labelsFolds{i};