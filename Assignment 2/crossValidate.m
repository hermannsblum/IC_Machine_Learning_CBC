function confusionMatrix = crossValidate(T, xvalues, labels, k, stratified)
% Returns the confusion matrix computed via cross-validation.
% T is the set of binary trees.
% xvalues are the feature values of the data set.
% labels are the class values associated to each data point.
% k is the number of folds.
% stratified is a Boolean value that specifies whether or not
% stratification of folds should be performed.

if(stratified)
    % Obtain the samples divided by class label
    samplesPerClass = stratifySample(xvalues,labels);
    % Split dataset in k folds
    folds = cell(k,1);
    labelsFolds = cell(k,1);
    for i=1:k
        % Perform the division in k folds in every class. Merge the
        % subfolds from all the classes to obtain a single fold
        for j=1:length(samplesPerClass)
            % Repmat repeats the class value the specified number of
            % times
            folds{i} = [folds{i}; getFold(samplesPerClass{j},repmat(j,size(samplesPerClass{j},1),1),k,i)];
            labelsFolds{i} = [labelsFolds{i}; repmat(j,size(folds{i},1)-size(labelsFolds{i},1),1)];
        end
        
    end
    
else

end

end