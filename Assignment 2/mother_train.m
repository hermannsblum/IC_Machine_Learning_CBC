function [ mother, trees] = mother_train( examples, attributes, labels )
%Train 6 binary trees on the training set (examples, labels) and return
%them into the tree array T.

[m, n] = size(examples); 
binary_targets = zeros(m, 1);

for tree = 1:6
   
    %Switch to binary labels
    index = find(labels == tree);
    binary_targets(index) = 1;
    
    %Train a tree to learn an emotion
    trees(tree) = DTTrain(examples, attributes, binary_targets);
    
    %DrawDecisionTree(T(i));
    
    binary_targets = zeros(m, 1);
    
end

% decisions of the trees on the training data are samples for the mother
decisions = zeros(m, 6);
for tree = 1:6
    for i = 1:m
        decisions(i, tree) = simple_prediction(trees(tree), examples(i, :));
    end
end

% train the mother
mother = meta_train(decisions, 1:6, labels);

%DrawDecisionTree(mother, 'Meta Tree');

end


function [tree] = meta_train(examples, classes, targets)
% train a "mother-tree" on deciding which tree gives the best decision

tree.op = [];
tree.kids = [];
tree.class = [];

if (sample_entropy(targets) == 0)
    % pure targets
    
    tree.class = maj_value(targets);
    
elseif isempty(classes)  
    % no more data to base decision on
    
    tree.class = maj_value(targets);
    
else
    
    tree.op = choose_best_attr(examples, classes, targets);
    tree.kids = cell(1, 2);
        
    % raise up the kids
    
    for j = 1:2
        
        child_index = find(examples(:, tree.op) == (j - 1));
        child_examples = examples(child_index, :);     
        child_targets = targets(child_index);
        
        if (isempty(child_examples))
            % we can't train this child, make a leaf with the majority
            % value of all training data coming to the parent
            
            tree.kids{j}.op = [];
            tree.kids{j}.kids = [];
            tree.kids{j}.class = maj_value(targets);
            
        else 
            
            % remove the used attribute from the list
            index = find(classes == tree.op);
            classes(index) = [];
            
            % recursive training with this child
            
            tree.kids{j} = meta_train(child_examples, classes, ...
                child_targets);
                    
        end
        
    end
    
end

end