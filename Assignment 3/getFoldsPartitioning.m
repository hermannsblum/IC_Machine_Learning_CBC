function foldsIndices = getFoldsPartitioning(labels, k, stratified)
% Returns the confusion matrix computed via cross-validation.
% T is the set of binary trees.
% xvalues are the feature values of the data set.
% labels are the class values associated to each data point.
% k is the number of folds.
% stratified is a Boolean value that specifies whether or not
% stratification of folds should be performed.

% Cell array containing the folds
foldsIndices = cell(k,1);
numClasses = length(unique(labels));

if(stratified)
    % Obtain the example indices divided by class label
    indicesPerClass = stratifySampleIndexed(labels);
    
    for i=1:k
        % Perform the division in k folds in every class. Merge the
        % subfolds from all the classes to obtain a single fold
        for j=1:numClasses
            foldsIndices{i} = [foldsIndices{i}; getFoldIndexed(indicesPerClass{j},k,i)];
        end
        foldsIndices{i} = sort(foldsIndices{i});
    end    
    
else
    % Non-stratified cross-validation
    for i=1:k
        foldsIndices{i} = getFoldIndexed(1:length(labels),k,i);
    end
end