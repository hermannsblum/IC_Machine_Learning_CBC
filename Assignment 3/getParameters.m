function [ parameters, numParams ] = getParameters( algorithm )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

neuronsPerLayer = 6:45;
hiddenLayers = 1:2;
% neuronsPerLayer = 2:6;
% hiddenLayers = 1:4;

switch algorithm
    case 'traingd'
        learningRates = [5 3 1 0.5 0.3 0.1 0.05 0.03 0.01];
        
        parameters = cell(3,1);
        parameters{1} = neuronsPerLayer;
        parameters{2} = hiddenLayers;
        parameters{3} = learningRates;
        
        numParams = zeros(1,3);
        for i=1:3
            numParams(i) = length(parameters{i});
        end
        
    case 'traingda'
        learningRates = [1 0.1 0.01];
        lrDecreaseRatios = [0.7 0.07 0.03];
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
        
        numParams = zeros(1,5);
        for i=1:5
            numParams(i) = length(parameters{i});
        end
        
    case 'traingdm'
        learningRates = [5 3 1 0.5 0.3 0.1 0.05 0.03 0.01];
        momenta = [0.9 0.95];
        
        parameters = cell(3,1);
        parameters{1} = neuronsPerLayer;
        parameters{2} = hiddenLayers;
        parameters{3} = learningRates;
        parameters{4} = momenta;
        
        numParams = zeros(1,4);
        for i=1:4
            numParams(i) = length(parameters{i});
        end
        
    case 'trainrp'
        delta_inc = [1.4 1.3 1.2 1.1]; % needs to be greater than 1
        delta_dec = [0.7 0.5 0.3 0.07 0.05 0.03];
        
        parameters = cell(4,1);
        parameters{1} = neuronsPerLayer;
        parameters{2} = hiddenLayers;
        parameters{3} = delta_inc;
        parameters{4} = delta_dec;
        
        numParams = zeros(1,4);
        for i=1:4
            numParams(i) = length(parameters{i});
        end

end

