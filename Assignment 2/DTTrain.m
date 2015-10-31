function [ tree ] = DTTrain( examples, attributes, binary_targets )
%Train a decision tree on the dataset (examples, binary_target)
%according to the ID3 algorithm. The examples have the attributes listed in
%attributes

%Initialize an empty tree
tree.op = [];
%tree.kids = cell(1, 2);
tree.kids = [];
tree.class = [];

p = length(find(binary_targets == 1)); %Number of positives
n = length(find(binary_targets == 0)); %Number of negatives

if (sampleEntropy(p, n) == 0)   %Check the purity
    
    %fprintf('Leaf: pure entrpy \n');
    %fprintf('Program paused. Press enter to continue.\n');
    %pause;
    
    tree.class = majValue(binary_targets);
    
elseif (isempty(attributes))    %Check if no more attributes are avilable
    
    %fprintf('Leaf: no more attribute \n');
    %fprintf('Program paused. Press enter to continue.\n');
    %pause;
    
    tree.class = majValue(binary_targets);
    
else
    
    tree.op = chooseBestAtt(examples, attributes, binary_targets);
    tree.kids = cell(1, 2);
        
    %tree
    
    for j = 1:2
        
        %fprintf('#Examples: %d \n', length(examples));
        xi_index = find(examples(:, tree.op) == (j - 1));
        xi_ex = examples(xi_index, :);     
        xi_binary_targets = binary_targets(xi_index);
        
        %fprintf('Attribute selected : %d \n', tree.op);
        %fprintf('#Examples with xi = 0: %d \n', length(xi_index));
        %fprintf('Program paused. Press enter to continue.\n');
        %pause;
        
        if (isempty(xi_ex))
       
            %fprintf('Leaf: pure entropy \n');
            %fprintf('Program paused. Press enter to continue.\n');
            %pause;
            
            tree.kids{j}.op = [];
            tree.kids{j}.kids = [];
            tree.kids{j}.class = majValue(binary_targets);
            
        else 
            
            %fprintf('Add a subtree. \n', tree.op);
            %fprintf('Program paused. Press enter to continue.\n');
            %pause;
            
            index = find(attributes == tree.op);
            attributes(index) = [];
            
            tree.kids{j} = DTTrain(xi_ex, attributes, xi_binary_targets);
                    
        end
        
    end
    
end

end

