function trSIndices = getTrainingSetIndexed(indices,i)
% Returns the training set for the i-th iteration of cross-validation.
% It is just a concatenation of all the folds except the i-th

trSIndices = [];
for j=1:length(indices)
    if(j~=i)
        trSIndices = [trSIndices; indices{j}];
    end
end