function [ tree ] = DTTrain( examples, attributes, binary_targets)
    tree = significance_training(examples, attributes, binary_targets, 1);

end

function [tree] = significance_training(examples, attributes, ...
    binary_targets, significance)
%Train a decision tree on the dataset (examples, binary_target)
%according to the ID3 algorithm. The examples have the attributes listed in
%attributes

%Initialize an empty tree
tree.op = [];
tree.kids = [];
tree.class = [];
tree.p_error = 0;
tree.significance = significance;

if (sample_entropy(binary_targets) == 0)
    % pure targets
    
    tree.class = maj_value(binary_targets);
    tree.p_error = 0;
    
elseif isempty(attributes)  
    % no more attributes to base decision on
    
    tree.class = maj_value(binary_targets);
    n_correct_targets = length(find(binary_targets == tree.class));
    tree.p_error = n_correct_targets / length(binary_targets);
    tree.significance = tree.p_error * significance;
    
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
            tree.kids{j}.significance = 0;
            tree.kids{j}.p_error = 1;
            
        else 
            
            % remove the used attribute from the list
            index = find(attributes == tree.op);
            attributes(index) = [];
            
            % recursive training with this child
            child_significance = length(child_binary_targets) / ...
                length(binary_targets) * significance;
            
            tree.kids{j} = significance_training(child_examples, attributes, ...
                child_binary_targets, child_significance);
                    
        end
        
    end
    
end

end

