function [ T ] = train( examples, attributes, labels )
%Train 6 binary trees on the training set (examples, labels) and return
%them into the tree array T.

[m, n] = size(examples); 
binary_targets = zeros(m, 1);

for i = 1:6
   
    %Switch to binary labels
    index = find(labels == i);
    binary_targets(index) = 1;
    
    %Train a tree to learn an emotion
    T(i) = DTTrain(examples, attributes, binary_targets);
    
    DrawDecisionTree(T(i));
    
    binary_targets = zeros(m, 1);
    
end

end