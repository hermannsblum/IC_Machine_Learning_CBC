function [foldAttrs, foldLabels] = getFold(examples, labels, k, i)
% Get the i-th fold in a partition by k of the examples set

numExamples = size(examples,1);
% If the number of examples is less than the number of folds, return only
% one example for i<=numExamples and no samples for i>numExamples
if(k>numExamples)
    if(i<numExamples)
        foldAttrs = examples(i,:);
        foldLabels = labels(i);
    else
        foldAttrs = [];
        foldLabels = [];
    end
    
else
    sizeFold = floor(numExamples/k);
    remainder = mod(numExamples, k);


    % If numExamples is not multiple of k, the first folders have one example
    % more than the others
    if(i<=remainder)
        sizeFold = sizeFold+1;
    end

    startIndex = (i-1)*sizeFold+1;
    if(i>remainder)
        startIndex = startIndex+remainder;
    end
    endIndex = startIndex+sizeFold-1;

    foldAttrs = examples(startIndex:endIndex,:);
    foldLabels = labels(startIndex:endIndex);
end