function samplesPerClass = stratifySample(xvalues,labels)
% Returns a cell array of matrices, where each matrix contains all the
% sample points with the same label.
% Works only with integer labels.
% Used for stratified cross-validation.

samplesPerClass = cell(length(labels));

for i=1:length(labels)
    % In the i-th position of the cell array, put the samples whose label
    % is i.
    samplesPerClass{i} = xvalues(labels==i,:);
end

end