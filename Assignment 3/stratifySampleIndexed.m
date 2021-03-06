function samplesPerClass = stratifySampleIndexed(labels)
% Returns a cell array of matrices, where each matrix contains the indices 
% of all the sample points with the same label.
% Works only with integer labels.
% Used for stratified cross-validation.
samplesPerClass = cell(length(unique(labels)),1);
for i=1:length(unique(labels))
% In the i-th position of the cell array, put the samples whose label
% is i.
samplesPerClass{i} = find(labels==i)';
end
end