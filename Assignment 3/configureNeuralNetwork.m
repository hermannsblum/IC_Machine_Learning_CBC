function [net] = configureNeuralNetwork(algorithm, parameters)

% set common parameters
neuronsPerLayer = parameters(1);
hiddenLayers = parameters(2);

switch algorithm
    case 'traingd'
        net.trainParam.lr = parameters(3);
    case 'traingda'
        net.trainParam.lr = parameters(3);
        net.trainParam.lr_dec = parameters(4);
        net.trainParam.lr_inc = parameters(5);
    case 'traingdm'
        net.trainParam.lr = parameters(3);
        net.trainParam.mc = parameters(4);
    case 'trainrp'
    % Define the candidate parameter values
        delt_inc = parameters(3);
        delt_dec = parameters(4);
        
        net = feedforwardnet(repmat(neuronsPerLayer,1,hiddenLayers),algorithm);
        net.trainParam.delt_inc = delt_inc;
        net.trainParam.delt_dec = delt_dec;
        
end
