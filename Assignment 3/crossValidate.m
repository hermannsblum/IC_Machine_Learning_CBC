function [ parameters, accuracies ] = crossValidate( algorithm, attributes, labels )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

neuronsPerLayer = 6:45;
hiddenLayers = 1:3;

% Get the indices of the 10 folds as a cell array of 10 indices arrays
foldIndices = getFoldsPartitioning(labels,10,true);
[attributesNN,labelsNN] = ANNdata(attributes,labels);

switch algorithm
    case 'traingd'
        
    case 'traingda'
        % Define the candidate parameter values
        learningRates = [5 2 1 0.5 0.2 0.1 0.05 0.02 0.01];
        delta_inc = [0.7 0.5 0.07 0.05];
        delta_dec = [1.4 2 5];
        
        parameters = cell(5,1);
        parameters{1} = hiddenLayers;
        parameters{2} = neuronsPerLayer;
        parameters{3} = learningRates;
        parameters{4} = delta_inc;
        parameters{5} = delta_dec;
        
        accuracies = zeros(length(hiddenLayers),...
            length(neuronsPerLayer),...
            length(learningRates),...
            length(delta_inc),...
            length(delta_dec));
        
        for a = 1:length(hiddenLayers)
            l = hiddenLayers(a);
            for b = 1:length(neuronsPerLayer);
                npl = neuronsPerLayer(b);
                % Create a NN with l layers and npl neurons
                % per layer
                net = feedforwardnet(repmat(npl,1,l),algorithm);
                for c = 1:length(learningRates);
                    lr = learningRates(c);
                    % Set learning rate
                    net.trainParam.lr = lr;
                    for d = 1:length(delta_inc);
                        delt_dec = delta_inc(d);
                        % Set learning rate's decrement rate
                        net.trainParam.lr_dec = delt_dec;
                        for e = 1:length(delta_dec);
                            lr_inc = delta_dec(e);
                            % Set learning rate's increment rate
                            net.trainParam.lr_inc = lr_inc;
                            
                            accuracyPerFold = zeros(10,5);
                            for i = 1:10 % Loop over the folds
                                for j = 1:5 
                                    % Get training and validation indices
                                    trainingSetIndices = getTrainingSetIndexed(foldIndices,i);
                                    validationSetIndices = foldIndices{i};
                                    % Set indices for training and validation
                                    % sets
                                    net.divideFcn = 'divideind';
                                    net.divideParam.trainInd = trainingSetIndices;
                                    net.divideParam.valInd = validationSetIndices;
                                    net.divideParam.testInd = [];


                                    % TODO: Set overfitting avoidance
                                    % parameters

                                    % Set up input and output layer
                                    net = configure(net, attributesNN, labelsNN);
                                    % Train network
                                    net = train(net, attributesNN, labelsNN);
                                    % Get performance on validation set
                                    predictions = NNout2labels(sim(net, attributesNN(:,validationSetIndices)));
                                    confMatrix = getConfusionMatrix(labels(validationSetIndices),predictions,6);
                                    accuracyPerFold(i,j) = sum(diag(confMatrix))/length(predictions);
                                end
                            end
                            % Compute average accuracy for the current
                            % parameter configuration
                            accuracies(a,b,c,d,e) = mean(mean(accuracyPerFold,2),1);
                            
                        end
                    end
                end
            end
        end
        
        
    case 'traingdm'
        
    case 'trainrp'
        % Define the candidate parameter values
        learningRates = [5 2 1 0.5 0.2 0.1 0.05 0.02 0.01];
        delta_inc = [1.5 1.2 1.0 0.15 0.12 0.1];
        delta_dec = [0.7 0.5 0.3 0.07 0.05 0.03];
        
        parameters = cell(4,1);
        parameters{1} = hiddenLayers;
        parameters{2} = neuronsPerLayer;
        parameters{3} = delta_inc;
        parameters{4} = delta_dec;
        
        accuracies = zeros(length(hiddenLayers),...
            length(neuronsPerLayer),...
            length(delta_inc),...
            length(delta_dec));
        
        for a = 1:length(hiddenLayers)
            l = hiddenLayers(a);
            for b = 1:length(neuronsPerLayer);
                npl = neuronsPerLayer(b);
                % Create a NN with l layers and npl neurons
                % per layer
                net = feedforwardnet(repmat(npl,1,l),algorithm);
                for c = 1:length(delta_inc);
                    delt_inc = delta_inc(c);
                    % Set delta inc
                    net.trainParam.delt_inc = delt_inc;
                    for d = 1:length(delta_dec);
                        delt_dec = delta_dec(d);
                        % Set delta dec
                        net.trainParam.delt_dec = delt_dec;

                        accuracyPerFold = zeros(10,5);
                        for i = 1:10 % Loop over the folds
                            for j = 1:5 
                                % Get training and validation indices
                                trainingSetIndices = getTrainingSetIndexed(foldIndices,i);
                                validationSetIndices = foldIndices{i};
                                % Set indices for training and validation
                                % sets
                                net.divideFcn = 'divideind';
                                net.divideParam.trainInd = trainingSetIndices;
                                net.divideParam.valInd = validationSetIndices;
                                net.divideParam.testInd = [];


                                % TODO: Set overfitting avoidance
                                % parameters

                                % Set up input and output layer
                                net = configure(net, attributesNN, labelsNN);
                                % Train network
                                net = train(net, attributesNN, labelsNN);
                                % Get performance on validation set
                                predictions = NNout2labels(sim(net, attributesNN(:,validationSetIndices)));
                                confMatrix = getConfusionMatrix(labels(validationSetIndices),predictions,6);
                                accuracyPerFold(i,j) = sum(diag(confMatrix))/length(predictions);
                            end
                        end
                        % Compute average accuracy for the current
                        % parameter configuration
                        accuracies(a,b,c,d) = mean(mean(accuracyPerFold,2),1);

                        
                    end
                end
            end
        end
   % end trainrp
end





end

