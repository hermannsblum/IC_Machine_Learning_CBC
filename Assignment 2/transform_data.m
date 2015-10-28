function [ bin_target ] = transform_data( y, emo )
%TRANSFORM_DATA Filters data into binary vector with respect
%to the target emotion emo

bin_target = (y == emo * ones(size(y)));

end

