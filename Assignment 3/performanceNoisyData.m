load('noisydata_students.mat');
% Convert data to NN format
[xNN, yNN] = ANNdata(x,y);

foldsIndices = getFoldsPartitioning(y,10,true);
for i=1:10
    % Split: Test- 1 Fold (i) Validation- 1 Fold (k) Training-8
    testSetIndices = foldsIndices{i};
    k = mod(i,10)+1;
    validationSetIndices = foldsIndices{k};
    trainingSetIndices = [];
    for j=1:10
        if j~=i && j~=k
            trainingSetIndices = [trainingSetIndices foldsIndices{j}];
        end
    end
    
    % The file confMatrixFold{i}.mat contains the optimal network computed
    % on the clean data. The parameters are already set. All we have to do
    % is retrain the network on the noisy data
    load(['confMatrixFold' num2str(i) '.mat']);

    % To be consistent with the rest of the exercise, train a network five
    % times and take the one that performs best on the validation set
    [mserror,net] = repeatNNTraining(net,xNN,yNN,trainingSetIndices,validationSetIndices);
    
    % Compute and save confusion matrix on test fold
    predictionsNN = sim(net,xNN(:,testSetIndices));
    predictions = NNout2labels(predictionsNN);
    confMatrixNoisyFold = getConfusionMatrix(y(testSetIndices),predictions,6);
    save(['confMatrixNoisyFold' num2str(i)],'net','confMatrixNoisyFold');
end