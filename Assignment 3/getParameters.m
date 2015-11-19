function [ parameters, numParams ] = getParameters( algorithm )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

neuronsPerLayer = 6:45;
hiddenLayers = 1:3;
% neuronsPerLayer = 2;
% hiddenLayers = 1;

switch algorithm
    case 'traingd'
        
    case 'traingda'
        learningRates = [5 2 1 0.5 0.2 0.1 0.05 0.02 0.01];
        lrDecreaseRatios = [0.7 0.5 0.07 0.05];
        lrIncreaseRatios = [1.4 2 5];
%         learningRates = 0.5;
%         lrDecreaseRatios = [0.7 0.5];
%         lrIncreaseRatios = [1.4 2];

        parameters = cell(5,1);
        parameters{1} = neuronsPerLayer;
        parameters{2} = hiddenLayers;
        parameters{3} = learningRates;
        parameters{4} = lrDecreaseRatios;
        parameters{5} = lrIncreaseRatios;
        
        numParams = 5;
        for i=1:5
            numParams(i) = length(parameters{i});
        end
        
    case 'traingdm'
        
    case 'trainrp'
        delta_inc = [1.5 1.2 1.0 0.15 0.12 0.1];
        delta_dec = [0.7 0.5 0.3 0.07 0.05 0.03];
        
        parameters = cell(4,1);
        parameters{1} = hiddenLayers;
        parameters{2} = neuronsPerLayer;
        parameters{3} = delta_inc;
        parameters{4} = delta_dec;

end

