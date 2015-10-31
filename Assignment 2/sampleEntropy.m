function [ E_s ] = sampleEntropy( pos, neg )
%sampleEntropy computes and return in E_s the entropy of a sample with pos
%positives and neg negatives

n_ex = pos + neg;

if (pos == 0 || neg == 0)
    
    E_s = 0; %The sample is already pure
    
else

    E_s = - pos/n_ex * log2(pos/n_ex) - neg/n_ex * log2(neg/n_ex);
    
end

end

