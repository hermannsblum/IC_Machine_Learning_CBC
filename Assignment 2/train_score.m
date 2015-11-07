function [ T ] = train_score( examples, attributes, labels )
%Train 6 binary trees on the training set (examples, labels) and return
%them into the tree array T.

[m, n] = size(examples); 
binary_targets = zeros(m, 1);

for i = 1:6
   
    %Switch to binary labels
    index = labels == i;
    binary_targets(index) = 1;
    
    %Train a tree to learn an emotion
    T(i) = DTTrain_score(examples, attributes, binary_targets);
    
    binary_targets = zeros(m, 1);
    
end

end