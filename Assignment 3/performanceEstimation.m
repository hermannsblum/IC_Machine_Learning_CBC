function [confusion, recall, precision, f1, accuracy] = ...
    performanceEstimation(attributes, labels)

algorithms = ['traingd', 'traingda', 'traingdm', 'trainrp'];
n_alg = length(algorithms);

% Get the indices of the 10 folds as a cell array of 10 indices arrays
foldIndices = getFoldsPartitioning(labels,10,true);
[attributesNN,labelsNN] = ANNdata(attributes,labels);

for i = 1:10 % iterate over 10 test folds
    % share: Test-1 Validation-1 Training-8
    testSetIndices = foldIndices{i};
    k = i - 1;
    if k == 0
        k = 10;
    end
    validationSetIndices = foldIndices{k};
    trainingSetIndices = getTrainingSetIndexed(...
        getTrainingSetIndexed(foldIndices, i), k);
    
    % find optimal Parameters with training and validation set
    optimalParameters = cell(n_alg);
    for k = 1:n_alg
        parameters = getParameters(algorithms(k));
        accuracy = validateNeuralNetwork(algorithms(k), parameters, ...
            attributesNN, labelsNN, trainingSetIndices, validationSetIndices);
        [val, idx] = max(accuracy(:));
        idxParameters = ind2sub(size(accuracy), idx);
        thisParameters = zeros(length(parameters));
        for j = 1:length(parameters)
            thisParameters(j) = parameters{j}(idxParameters(j));
        end
        optimalParameters{k} = thisParameters;
    end
    
    % train with optimal parameters on joined trainign and validation set
    trainingSetIndices = getTrainingSetIndexed(foldIndices, i);
    
    confusion = cell(n_alg, 1);
    recall = zeros(n_alg);
    precision = zeros(n_alg);
    f1 = zeros(n_alg);
    accuracy = zeros(n_alg);
    for k = 1:n_alg
        net = configureNeuralNetwork(algorithms(k), optimalParameters{k});
        
        % Set indices for training and test sets
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = trainingSetIndices;
        net.divideParam.valInd = [];
        net.divideParam.testInd = testSetIndices;
        
        % 5 iterations to reduce noise from randomness
        numIterations = 5;
        confusionMats = zeros(6,6);
        accuracies = zeros(numIterations);
        recalls = zeros(numIterations);
        precisions = zeros(numIterations);
        f1s = zeros(numIterations)
        for j = 1:numIterations
            % Set up input and output layer
            net = configure(net, attributesNN, labelsNN);
            % Train network
            net = train(net, attributesNN, labelsNN);
            % Get performance on testset
            predictions = NNout2labels(sim(net, attributesNN(:,testSetIndices)));
            confusionMat = getConfusionMatrix(labels(testSetIndices),predictions,6);

            % calculate statistics
            confusionMats = confusionMats + confusionMat;
            accuracies(j) = sum(diag(confusionMat))/length(predictions);
            
        
    
    
    
        
        
            
        