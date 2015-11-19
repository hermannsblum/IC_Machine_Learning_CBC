function [ accuracies ] = validateNeuralNetwork( algorithm, parameters, attributes, labels, trainingSetIndices, validationSetIndices )
% attributes and labels must be in CBC format (rows representing single
% examples, columns attributes; one-column labels)

[attributesNN,labelsNN] = ANNdata(attributes,labels);

% Set common parameters
neuronsPerLayer = parameters{1};
hiddenLayers = parameters{2};

switch algorithm
    case 'traingd'
        
    case 'traingda'
        % Define the candidate parameter values
        learningRates = parameters{3};
        lrDecreaseRatios = parameters{4};
        lrIncreaseRatios = parameters{5};
        
        accuracies = zeros(length(neuronsPerLayer),...
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
                            % From http://uk.mathworks.com/help/nnet/ug/improve-neural-network-generalization-and-avoid-overfitting.html?refresh=true
                            % Train different networks and take the one
                            % with the minimum mse to avoid overfitting
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
                            [~, jOpt] = min(perfs(j));
                            net = NN{jOpt};
                            % Get performance on validation set
                            predictions = NNout2labels(sim(net, attributesNN(:,validationSetIndices)));
                            confMatrix = getConfusionMatrix(labels(validationSetIndices),predictions,6);
                            
                            % Compute average accuracy for the current
                            % parameter configuration
                            accuracies(a,b,c,d,e) = sum(diag(confMatrix))/length(predictions);
                            
                        end
                    end
                end
            end
        end
        
        
    case 'traingdm'
        
    case 'trainrp'
    % Define the candidate parameter values
        delta_inc = parameters{3};
        delta_dec = parameters{4};
        
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

                        trialsAccuracies = zeros(1,5);
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
                            trialsAccuracies(j) = sum(diag(confMatrix))/length(predictions);
                            
                        end
                        % Compute average accuracy for the current
                        % parameter configuration
                        accuracies(a,b,c,d) = mean(trialsAccuracies);

                        
                    end
                end
            end
        end

end

