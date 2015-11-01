function trS = trainingSet(folds,i)
% Returns the training set for the i-th iteration of cross-validation

trS = [];
for j=1:length(folds)
    if(j~=i)
        trS = [trS; folds{j}];
    end
end