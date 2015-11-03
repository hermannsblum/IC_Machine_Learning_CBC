function [confusionMatrix, accuracy, precision, recall] = crossValidate(xvalues, labels, k, stratified)
% Returns the confusion matrix computed via cross-validation.
% T is the set of binary trees.
% xvalues are the feature values of the data set.
% labels are the class values associated to each data point.
% k is the number of folds.
% stratified is a Boolean value that specifies whether or not
% stratification of folds should be performed.

% Cell array containing the folds
folds = cell(k,1);
labelsFolds = cell(k,1);
numClasses = length(unique(labels));

if(stratified)
    % Obtain the samples divided by class label
    samplesPerClass = stratifySample(xvalues,labels);
    
    for i=1:k
        % Perform the division in k folds in every class. Merge the
        % subfolds from all the classes to obtain a single fold
        for j=1:numClasses
            % Repmat repeats the class value the specified number of
            % times. It creates an array of values all equal to j
            folds{i} = [folds{i}; getFold(samplesPerClass{j},repmat(j,size(samplesPerClass{j},1),1),k,i)];
            % Add as many labels as the number of examples added to
            % folds{i} in this iteration of the for loop. This number is
            % the current size of folds{i} minus the previous size of
            % folds{i} (which in turn is equal to the current size of
            % labelsFolds{i}).
            labelsFolds{i} = [labelsFolds{i}; repmat(j,size(folds{i},1)-size(labelsFolds{i},1),1)];
        end
        
    end
    
%     % Test if proportion of class values in each fold is equal
%     testProportions = zeros(k,6);
%     for i=1:k
%         for j=1:6
%             testProportions(i,j) = sum(labelsFolds{i}==j)/length(labelsFolds{i});
%         end
%     end
%     testProportions
%     sum(testProportions,2)
    
    
else
    % Non-stratified cross-validation
    for i=1:k
        [folds{i}, labelsFolds{i}] = getFold(xvalues,labels,k,i);
    end
end

% Perform cross-validation. Train on k-1 folds and test on the i-th
% fold. Rotate the testing fold.
predictions = cell(k,1);
% Confusion matrix over all the examples
confusionMatrix = zeros(numClasses);

accuracyFolds = zeros(k,1);
precisionFolds = zeros(k,numClasses);
recallFolds = zeros(k,numClasses);
fMeasureFolds = zeros(k,numClasses);

for i=1:k
    [trainingSet, labelsTraining] = getTrainingSet(folds,labelsFolds,i);
    [testSet, labelsTest] = getTestSet(folds,labelsFolds,i);
    T = train(trainingSet,1:45,labelsTraining);
    predictions{i} = testTreesRandomChoice(T,testSet);
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

end

accuracy = mean(accuracyFolds);
precision = mean(precisionFolds,1);
recall = mean(recallFolds,1);

end