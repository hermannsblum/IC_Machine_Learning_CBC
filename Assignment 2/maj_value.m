function [ value ] = maj_value( binary_targets )
%majVote computes and return in value the mode of the set of labels 
%binary_target provided as input

value = mode(binary_targets);

end

