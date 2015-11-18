function [confusionMatrix, accuracy, precision, recall, fmeasure] = crossValidateDecisionTree(xvalues, labels, k, stratified)
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

% Perform cross-validation. Train on k-1 folds and test on the i-th
% fold. Rotate the testing fold.
predictions = cell(k,1);
% Confusion matrix over all the examples in all the folds
confusionMatrix = zeros(numClasses);

% Performance metrics per fold
accuracyFolds = zeros(k,1);
precisionFolds = zeros(k,numClasses);
recallFolds = zeros(k,numClasses);
fMeasureFolds = zeros(k,numClasses);

for i=1:k
    trainingSetIndices = getTrainingSetIndexed(foldsIndices,i);
    testSetIndices = getTestSetIndexed(foldsIndices,i);
    labelsTest = labels(foldsIndices{i});
    %T = train_score(trainingSet,1:45,labelsTraining);
    %predictions{i} = decide_by_score(T,testSet);
    T = train_score(xvalues(trainingSetIndices,:), 1:45, labels(trainingSetIndices));
    predictions{i} = decide_by_score(T, xvalues(testSetIndices,:));
    % Confusion matrix over a single fold
    confusionMatrixFold = zeros(numClasses);
    for j=1:length(predictions{i})
        confusionMatrixFold(labelsTest(j),predictions{i}(j)) = confusionMatrixFold(labelsTest(j),predictions{i}(j))+1;
    end

    confusionMatrix = confusionMatrix+confusionMatrixFold;
    accuracyFolds(i) = sum(diag(confusionMatrixFold))/sum(sum(confusionMatrixFold));

    % After computing the confusion matrix of the single fold, the TPs
    % of each class are in the diagonal of such matrix, while the
    % values TP+FP are the column sums.
    precisionFolds(i,:) = diag(confusionMatrixFold)./sum(confusionMatrixFold,1)';
    recallFolds(i,:) = diag(confusionMatrixFold)./sum(confusionMatrixFold,2);
    
    % If both precision and recall are 0, the corresponding F-measure is 0,
    % otherwise it is computed according to the formula.
    for j=1:numClasses
        if(precisionFolds(i,j)+recallFolds(i,j)==0)
            fMeasureFolds(i,j)=0;
        else
            fMeasureFolds(i, j) = 2* precisionFolds(i,j).*recallFolds(i,j) ...
                ./ (precisionFolds(i,j) + recallFolds(i,j));
        end
    end

end

% Average performance metrics across the folds
accuracy = mean(accuracyFolds);
precision = mean(precisionFolds,1);
recall = mean(recallFolds,1);
fmeasure = mean(fMeasureFolds,1);
end