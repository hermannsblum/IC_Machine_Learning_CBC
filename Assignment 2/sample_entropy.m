function [ E ] = sample_entropy( sample )
%ENTROPY of positive and negative examples of a set of training data y

n = length(sample);
pos = sum(sample);
neg = n - pos;

if (pos == 0 || neg == 0)
    
    E = 0; %The sample is already pure
    
else
    
    E = - pos/n * log2(pos/n) - neg/n * log2(neg/n);
    
end

end

