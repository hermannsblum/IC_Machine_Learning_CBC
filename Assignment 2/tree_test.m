load('cleandata_students.mat')
% choose emotion
emo = 1;
% filter targets
bin_target = (y == emo * ones(size(y)));

% generate list of attributes
attributes = 1:size(x, 2)

% make a tree
decision_tree = DTTrain (x, attributes, bin_target);

DrawDecisionTree(decision_tree, 'My Tree');