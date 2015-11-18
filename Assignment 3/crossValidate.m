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
        lrDecreaseRatios = [0.7 0.5 0.07 0.05];
        lrIncreaseRatios = [1.4 2 5];
        
        parameters = cell(5,1);
        parameters{1} = hiddenLayers;
        parameters{2} = neuronsPerLayer;
        parameters{3} = learningRates;
        parameters{4} = lrDecreaseRatios;
        parameters{5} = lrIncreaseRatios;
        
        accuracies = zeros(length(hiddenLayers),...
            length(neuronsPerLayer),...
            length(learningRates),...
            length(lrDecreaseRatios),...
            length(lrIncreaseRatios));
        
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
                    for d = 1:length(lrDecreaseRatios);
                        lr_dec = lrDecreaseRatios(d);
                        % Set learning rate's decrement rate
                        net.trainParam.lr_inc = lr_inc;
                        for e = 1:length(lrIncreaseRatios);
                            lr_inc = lrIncreaseRatios(e);
                            % Set learning rate's increment rate
                            net.trainParam.lr_inc = lr_inc;
                            
                            accuracyPerFold = zeros(10,1);
                            for i = 1:10 % Loop over the folds
                                % Get training and validation indices
                                trainingSetIndices = getTrainingSetIndexed(foldIndices,i);
                                validationSetIndices = foldIndices{i};
                                % Set indices for training and validation
                                % sets
                                net.divideFcn = 'divideInd';
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
                                predictions = NNout2labels(sim(net, attributes(:,validationSetIndices)));
                                confMatrix = getConfusionMatrix(labels(validationSetIndices),predictions,6);
                                accuracyPerFold(i) = sum(diag(confMatrix))/length(predictions);
                            end
                            % Compute average accuracy for the current
                            % parameter configuration
                            accuracies(a,b,c,d,e) = mean(accuracyPerFold);
                            
                        end
                    end
                end
            end
        end
        
        
    case 'traingdm'
        
    case 'trainrp'
        
end





end

