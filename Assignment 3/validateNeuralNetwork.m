function [ mserrors ] = validateNeuralNetwork( algorithm, parameters, attributes, labels, trainingSetIndices, validationSetIndices )
% attributes and labels must be in CBC format (rows representing single
% examples, columns attributes; one-column labels)

[attributesNN,labelsNN] = ANNdata(attributes,labels);

% Set common parameters
neuronsPerLayer = parameters{1};
hiddenLayers = parameters{2};

switch algorithm
    case 'traingd'
        % Define the candidate parameter values (learning rates)
        learningRates = parameters{3};
        mserrors = zeros(length(neuronsPerLayer), length(hiddenLayers), length(learningRates));
        
        for a = 1:length(neuronsPerLayer)
            npl = neuronsPerLayer(a);
            for b = 1:length(hiddenLayers);
                l = hiddenLayers(b);
                % Create a NN with l layers and npl neurons
                % per layer
                net = feedforwardnet(repmat(npl,1,l),algorithm);
                % To avoid showing the performance window
                net.trainParam.showWindow = 0;
                for c = 1:length(learningRates);
                    lr = learningRates(c);
                    % Set learning rate
                    net.trainParam.lr = lr;
                    
                    disp(['Testing parameters npl=' num2str(npl)...
                                ' l=' num2str(l) ' lr=' num2str(lr)]);

                    mserrors(a,b,c) = findMinimumError(net, ...
                                attributesNN, labelsNN, trainingSetIndices, ...
                                validationSetIndices);
                    
                end
                
            end
            
        end
        
    case 'traingda'
        % Define the candidate parameter values
        learningRates = parameters{3};
        lrDecreaseRatios = parameters{4};
        lrIncreaseRatios = parameters{5};
        
        mserrors = zeros(length(neuronsPerLayer),...
            length(hiddenLayers),...
            length(learningRates),...
            length(lrDecreaseRatios),...
            length(lrIncreaseRatios));
        
        for a = 1:length(neuronsPerLayer)
            npl = neuronsPerLayer(a);
            for b = 1:length(hiddenLayers);
                l = hiddenLayers(b);
                % Create a NN with l layers and npl neurons
                % per layer
                net = feedforwardnet(repmat(npl,1,l),algorithm);
                % To avoid showing the performance window
                net.trainParam.showWindow = 0;
                for c = 1:length(learningRates);
                    lr = learningRates(c);
                    % Set learning rate
                    net.trainParam.lr = lr;
                    for d = 1:length(lrDecreaseRatios);
                        lr_dec = lrDecreaseRatios(d);
                        % Set learning rate's decrement rate
                        net.trainParam.lr_dec = lr_dec;
                        for e = 1:length(lrIncreaseRatios);
                            lr_inc = lrIncreaseRatios(e);
                            % Set learning rate's increment rate
                            net.trainParam.lr_inc = lr_inc;
                            
                            disp(['Testing parameters npl=' num2str(npl)...
                                ' l=' num2str(l) ' lr=' num2str(lr)...
                                ' lr_dec=' num2str(lr_dec) ' lr_inc=' num2str(lr_inc)]);

                            mserrors(a,b,c,d,e) = findMinimumError(net, ...
                                attributesNN, labelsNN, trainingSetIndices, ...
                                validationSetIndices);
                            
                        end
                    end
                end
            end
        end
        
        
    case 'traingdm'
        learningRates = parameters{3};
        momenta = parameters{4};
        
        mserrors = zeros(length(neuronsPerLayer),...
            length(hiddenLayers),...
            length(learningRates),...
            length(momenta));
        
        for a = 1:length(neuronsPerLayer)
            npl = neuronsPerLayer(a);
            for b = 1:length(hiddenLayers);
                l = hiddenLayers(b);
                % Create a NN with l layers and npl neurons
                % per layer
                net = feedforwardnet(repmat(npl,1,l),algorithm);
                % To avoid showing the performance window
                net.trainParam.showWindow = 0;
                for c = 1:length(learningRates);
                    lr = learningRates(c);
                    % Set learning rate
                    net.trainParam.lr = lr;
                    for d = 1:length(momenta);
                        mc = momenta(d);
                        % Set momentum
                        net.trainParam.mc = mc;
                        
                        disp(['Testing parameters npl=' num2str(npl)...
                                ' l=' num2str(l) ' lr=' num2str(lr)...
                                ' mc=' num2str(mc)]);
                        
                        mserrors(a, b, c, d) = findMinimumError(net, ...
                            attributesNN, labelsNN, trainingSetIndices, ...
                            validationSetIndices);
                    end
                end
            end
        end 
        
    case 'trainrp'
    % Define the candidate parameter values
        delta_inc = parameters{3};
        delta_dec = parameters{4};
        
        mserrors = zeros(length(neuronsPerLayer),...
            length(hiddenLayers),...
            length(delta_inc),...
            length(delta_dec));
        
        for a = 1:length(neuronsPerLayer)
            npl = neuronsPerLayer(a);
            for b = 1:length(hiddenLayers);
                l = hiddenLayers(b);
                % Create a NN with l layers and npl neurons
                % per layer
                net = feedforwardnet(repmat(npl,1,l),algorithm);
                % To avoid showing the performance window
                net.trainParam.showWindow = 0;
                for c = 1:length(delta_inc);
                    delt_inc = delta_inc(c);
                    % Set delta inc
                    net.trainParam.delt_inc = delt_inc;
                    for d = 1:length(delta_dec);
                        delt_dec = delta_dec(d);
                        % Set delta dec
                        net.trainParam.delt_dec = delt_dec;
                        
                        disp(['Testing parameters npl=' num2str(npl)...
                                ' l=' num2str(l) ' delt_inc=' num2str(delt_inc)...
                                ' delt_dec=' num2str(delt_dec)]);
                        
                        mserrors(a, b, c, d) = findMinimumError(net, ...
                            attributesNN, labelsNN, trainingSetIndices, ...
                            validationSetIndices);
                    end
                end
            end
        end

end
end


function [mserror] = findMinimumError(net, attributesNN, labelsNN, ...
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
[mserror] = min(perfs(j));
%net = NN{jOpt};

% % Get performance on validation set
% predictions = NNout2labels(sim(net, attributesNN(:,validationSetIndices)));
% confMatrix = getConfusionMatrix(labels(validationSetIndices),predictions,6);
% 
% % Compute average accuracy for the current
% % parameter configuration
% mserror = sum(diag(confMatrix))/length(predictions);
end
