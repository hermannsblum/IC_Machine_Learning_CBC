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

                    mserrors(a,b,c) = repeatNNTraining(net, ...
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

                            mserrors(a,b,c,d,e) = repeatNNTraining(net, ...
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
                        
                        mserrors(a, b, c, d) = repeatNNTraining(net, ...
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
                        
                        mserrors(a, b, c, d) = repeatNNTraining(net, ...
                            attributesNN, labelsNN, trainingSetIndices, ...
                            validationSetIndices);
                    end
                end
            end
        end

end
end