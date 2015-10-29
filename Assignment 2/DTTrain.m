function [ tree ] = DTTrain( examples, attributes, binary_targets )
%Train a decision tree on the dataset (examples, binary_target)
%according to the ID3 algorithm. The examples have the attributes listed in
%attributes

%Initialize an empty tree
tree.op = [];
%tree.kids = cell(1, 2);
tree.kids = [];
tree.class = [];

if (sample_entropy(binary_targets) == 0 || isempty(attributes))   
    % either pure targets or no more attributes to base decision on
    
    tree.class = maj_value(binary_targets);
    
else
    
    tree.op = choose_best_attr(examples, attributes, binary_targets);
    tree.kids = cell(1, 2);
        
    % raise up the kids
    
    for j = 1:2
        
        fprintf('#Examples: %d \n', length(examples));
        child_index = find(examples(:, tree.op) == (j - 1));
        child_examples = examples(child_index, :);     
        child_binary_targets = binary_targets(child_index);
        
        %fprintf('Attribute selected : %d \n', tree.op);
        fprintf('#Examples with xi = 0: %d \n', length(child_index));
        %fprintf('Program paused. Press enter to continue.\n');
        %pause;
        
        if (isempty(child_examples))
            % we can't train this child, make a leaf with the majority
            % value of all training data coming to the parent
            
            tree.kids{j}.op = [];
            tree.kids{j}.kids = [];
            tree.kids{j}.class = maj_value(binary_targets);
            
        else 
            
            % remove the used attribute from the list
            index = find(attributes == tree.op);
            attributes(index) = [];
            
            % recursive training with this child
            tree.kids{j} = DTTrain(child_examples, attributes, ...
                child_binary_targets);
                    
        end
        
    end
    
end

end

