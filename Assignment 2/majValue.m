function [ value ] = majValue( binary_targets )
%majVote computes and returns the mode of the set of labels (the most
%frequent value)
%binary_target provided as input

value = mode(binary_targets);

end

