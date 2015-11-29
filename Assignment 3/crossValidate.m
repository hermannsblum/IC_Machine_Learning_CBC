function [ parameters, mserrors ] = crossValidate( algorithm, attributes, labels )

% Get the indices of the 10 folds as a cell array of 10 indices arrays
foldsIndices = getFoldsPartitioning(labels,10,true);

[parameters, numParams] = getParameters(algorithm);

mserrors = zeros(numParams);
for i=1:10
    disp(['Testing fold ' num2str(i)]);
    trainingSetIndices = getTrainingSetIndexed(foldsIndices,i);
    validationSetIndices = foldsIndices{i};
    
    mserrorsPerFold = validateNeuralNetwork(algorithm,parameters,attributes,labels,trainingSetIndices,validationSetIndices);
    save([algorithm '_msErrorsFold' num2str(i) '.mat'],'parameters','mserrorsPerFold');
    
    mserrors = mserrors+mserrorsPerFold;
end
% Average the accuracies
mserrors = mserrors./10;
save([algorithm '_avgmsErrors.mat'],'parameters','mserrors');


end

