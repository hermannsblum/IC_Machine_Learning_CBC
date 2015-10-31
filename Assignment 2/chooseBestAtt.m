function [ bestAtt ] = chooseBestAtt( examples, attributes, binary_targets )
%chooseBestAtt computes and return in bestAtt the attribute in attributes
%which determine the maximum information gain for the set 
%(examples, binary_targets). Return -1 if the set is pure because no split
%is required

%Compute the sample entropy
n_ex = length(binary_targets);         %Dataset size
p = length(find(binary_targets == 1)); %Number of positives
n = length(find(binary_targets == 0)); %Number of negatives

I = 0;
n_att = length(attributes);
E = sampleEntropy(p, n); %Sample entropy
    
for i = 1:n_att
       
    one_ex = binary_targets(find(examples(:, attributes(i)) == 1));   %examples with xi = 1
    zero_ex = binary_targets(find(examples(:, attributes(i)) == 0));  %Examples with xi = 0
        
    p1 = length(find(one_ex == 1)); %Number of positives in the 'xi-one set'
    n1 = length(find(one_ex == 0)); %Number of negatives in the 'xi-one set'
    n_ex1 = p1 + n1;    %Number of examples with xi = 1
        
    p0 = length(find(zero_ex == 1)); %Number of positives in the 'xi-zero set'
    n0 = length(find(zero_ex == 0)); %Number of negatives in the 'xi-zero set'
    n_ex0 = p0 + n0;    %Number of examples with xi = 0
        
    E_partition = n_ex1/n_ex * sampleEntropy(p1, n1) + n_ex0/n_ex * sampleEntropy(p0, n0);
        
    I_A = E - E_partition;
        
    if (I_A >= I)
            
       I = I_A;
       bestAtt = attributes(i);
        
    end
        
end

end

