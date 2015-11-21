function [net] = configureNeuralNetwork(algorithm, parameters)

% set common parameters
neuronsPerLayer = parameters(1);
hiddenLayers = parameters(2);

switch algorithm
    case 'traingd'
        
    case 'traingda'
        
    case 'traingdm'
        
    case 'trainrp'
    % Define the candidate parameter values
        delta_inc = parameters(3);
        delta_dec = parameters(4);
        
        net = feedforwardnet(repmat(neuronsPerLayer,1,hiddenLayers),algorithm);
        net.trainParam.delt_inc = delt_inc;
        net.trainParam.delt_dec = delt_dec;
        
end
