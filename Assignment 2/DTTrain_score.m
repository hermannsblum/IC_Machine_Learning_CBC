function [ tree ] = DTTrain_score( examples, attributes, binary_targets)
% wrapper for score_training to set score to 1 for the start
    tree = score_training(examples, attributes, binary_targets, 1);

end

function [tree] = score_training(examples, attributes, ...
    binary_targets, score)
%Train a decision tree on the dataset (examples, binary_target)
%according to the ID3 algorithm. The examples have the attributes listed in
%attributes. The score is a measure of correctly classified training
%examples at each stage and will be assigned to the leafs

%Initialize an empty tree
tree.op = [];
tree.kids = [];
tree.class = [];
tree.score = score;

if (sample_entropy(binary_targets) == 0)
    % pure targets
    
    tree.class = maj_value(binary_targets);
    
elseif isempty(attributes)  
    % no more attributes to base decision on
    
    tree.class = maj_value(binary_targets);
    n_correct_targets = length(find(binary_targets == tree.class));
    p_correct = n_correct_targets / length(binary_targets);
    tree.score = p_correct * score;
    
else
    
    tree.op = choose_best_attr(examples, attributes, binary_targets);
    tree.kids = cell(1, 2);
        
    % raise up the kids
    
    for j = 1:2
        
        child_index = find(examples(:, tree.op) == (j - 1));
        child_examples = examples(child_index, :);     
        child_binary_targets = binary_targets(child_index);
        
        if (isempty(child_examples))
            % we can't train this child, make a leaf with the majority
            % value of all training data coming to the parent
            
            tree.kids{j}.op = [];
            tree.kids{j}.kids = [];
            tree.kids{j}.class = maj_value(binary_targets);
            tree.kids{j}.score = 0;
            
        else 
            
            % remove the used attribute from the list
            index = find(attributes == tree.op);
            attributes(index) = [];
            
            % recursive training with this child
            child_score = length(child_binary_targets) / ...
                length(binary_targets) * score;
            
            tree.kids{j} = score_training(child_examples, attributes, ...
                child_binary_targets, child_score);
                    
        end
        
    end
    
end

end

