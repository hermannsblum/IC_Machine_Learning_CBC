function [confusionMatrix, accuracy, precision, recall, f1] = performanceEstimation(attributes, labels)

algorithms = {'traingd', 'traingda', 'traingdm', 'trainrp'};
n_alg = length(algorithms);
confusionMatrix = zeros(6);
accuracyFolds = zeros(10,1);
precisionFolds = zeros(10,6);
recallFolds = zeros(10,6);
f1Folds = zeros(10,6);

% Get the indices of the 10 folds as a cell array of 10 indices arrays
foldIndices = getFoldsPartitioning(labels,10,true);

for i = 1:10 % iterate over 10 test folds
    % Split: Test- 1 Fold (i) Validation- 1 Fold (k) Training-8
    testSetIndices = foldIndices{i};
    k = mod(i,10)+1;
    validationSetIndices = foldIndices{k};
    trainingSetIndices = [];
    for j=1:10
        if j~=i && j~=k
            trainingSetIndices = [trainingSetIndices foldIndices{j}];
        end
    end
    
    % find optimal Parameters with training and validation set
    bestPerformingAlgorithm = 1;
    bestMSE = Inf;
    optimalParameters = [];
    
    for k = 1:n_alg
        parameters = getParameters(algorithms{k});
        
        disp(['Fold ' num2str(i) ' Algorithm: ' algorithms{k}]);
        
        mserrorsAlgorithm = validateNeuralNetwork(algorithms{k}, parameters, ...
            attributes, labels, trainingSetIndices, validationSetIndices);
        % idx is the index of the minimum in the linearized
        % mserrorsAlgorithm
        [minMSE, idx] = min(mserrorsAlgorithm(:));
        % Get the corresponding indices in the multidimensional array
        idxParameters = cell(1,length(parameters));
        [idxParameters{:}] = ind2sub(size(mserrorsAlgorithm), idx);
        
        if(minMSE<bestMSE)
            bestMSE = minMSE;
            bestPerformingAlgorithm = k;
            optimalParameters = zeros(length(parameters),1);
            for j=1:length(parameters)
                optimalParameters(j) = parameters{j}(idxParameters{j});
            end
        end
    end
    
    optimalAlgorithm = algorithms{bestPerformingAlgorithm};
    
    % Configure the best training algorithm with the optimal parameter
    % configuration
    net = configureNeuralNetwork(optimalAlgorithm,optimalParameters);
    % Train the network 5 times and get the best network
    [attributesNN,labelsNN] = ANNdata(attributes,labels);
    [~,net] = repeatNNTraining(net,attributesNN,labelsNN,trainingSetIndices,validationSetIndices);
    
    % Compute performance on test set
    predictions = NNout2labels(sim(net,attributesNN(:,testSetIndices)));
    confMatrixFold = getConfusionMatrix(labels(testSetIndices),predictions,6);
    save(['confMatrixFold' num2str(i)],'net','confMatrixFold');
    
    accuracyFolds(i) = sum(diag(confMatrixFold))/length(testSetIndices); % The sum of all the elements in confMatrixFold is
                                                                         % equal to the size of the test set
    precisionFolds(i,:) = diag(confMatrixFold)./sum(confMatrixFold,1)';
    recallFolds(i,:) = diag(confMatrixFold)./sum(confMatrixFold,2);
    for j=1:6
        if(precisionFolds(i,j)+recallFolds(i,j)==0)
            f1Folds(i,j)=0;
        else
            f1Folds(i, j) = 2* precisionFolds(i,j).*recallFolds(i,j) ...
                ./ (precisionFolds(i,j) + recallFolds(i,j));
        end
    end
    confusionMatrix = confusionMatrix+confMatrixFold;
end
    
accuracy = mean(accuracyFolds);
precision = mean(precisionFolds,1);
recall = mean(recallFolds,1);
f1 = mean(f1Folds,1); 
            
end

        