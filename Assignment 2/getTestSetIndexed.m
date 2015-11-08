function tsIndices = getTestSetIndexed(foldsIndices,i)
% Returns the test set for the i-th iteration of the cross-validation

tsIndices = foldsIndices{i};
