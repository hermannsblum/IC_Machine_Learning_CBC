function [ best_attr ] = choose_best_attr( examples, attributes, binary_targets )
%chooseBestAtt computes and return in bestAtt the attribute in attributes
%which determine the maximum information gain for the set 
%(examples, binary_targets). Return -1 if the set is pure because no split
%is required

%Compute the sample entropy
n_ex = length(binary_targets);         %Dataset size
E = sample_entropy(binary_targets); %Sample entropy

% initial values
best_gain = 0;
best_attr = attributes(1); % for now this is as good as anything
    
for i = 1:length(attributes)
       
    % examples where attribute is 1
    one_attr = binary_targets(examples(:, attributes(i)) == 1); 
    % examples where attribute is 0
    zero_attr = binary_targets(examples(:, attributes(i)) == 0);
        
    E_partition = length(one_attr)/n_ex * sample_entropy(one_attr) ...
        + length(zero_attr)/n_ex * sample_entropy(zero_attr);
        
    gain = E - E_partition;
        
    if (gain > best_gain)
            
       best_gain = gain;
       best_attr = attributes(i);
        
    end
        
end

end

