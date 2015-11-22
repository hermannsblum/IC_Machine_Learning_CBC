function [mserror,net] = repeatNNTraining(net, attributesNN, labelsNN, ...
    trainingSetIndices, validationSetIndices)
% train 5 networks with random start weights and  choose the one with the 
% minumum Error

% From http://uk.mathworks.com/help/nnet/ug/improve-neural-network-generalization-and-avoid-overfitting.html?refresh=true
% Train different networks and take the one
% with the minimum mse to avoid overfitting

% labels = NNout2labels(labelsNN);

NN = cell(5,1);
perfs = zeros(5,1);
for j = 1:5
    % Set indices for training and validation
    % sets
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = trainingSetIndices;
    net.divideParam.valInd = validationSetIndices;
    net.divideParam.testInd = [];

    % Set up input and output layer
    NN{j} = configure(net, attributesNN, labelsNN);
    % Train network
    [NN{j}, trainRecord] = train(NN{j}, attributesNN, labelsNN);
    % Evaluate mean squared error on validation
    % set. trainRecord.best_vperf contains the
    % optimal error found in validation set
    % during the training
    perfs(j) = trainRecord.best_vperf;
end
[mserror,jOpt] = min(perfs(j));
net = NN{jOpt};

% % Get performance on validation set
% predictions = NNout2labels(sim(net, attributesNN(:,validationSetIndices)));
% confMatrix = getConfusionMatrix(labels(validationSetIndices),predictions,6);
% 
% % Compute average accuracy for the current
% % parameter configuration
% mserror = sum(diag(confMatrix))/length(predictions);
end
