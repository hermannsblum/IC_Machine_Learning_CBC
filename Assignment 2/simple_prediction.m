function [ class ] = simple_prediction( tree, sample )
%prediction_with_score walks along the tree to find the classification of
%the istance x and returns in addition a score calculated from the
%significance and probability of error of the leaf
    
    if (isempty(tree.op))   %Check if it is a leaf
        
        class = tree.class;
        
    else                    %Test the attribute
        
        sample_to_test = sample(tree.op);
        
        if (sample_to_test == 0) %Follow the left branch
           
           class = simple_prediction(tree.kids{1}, sample);
            
        else %Follow the right branch
            
           class= simple_prediction(tree.kids{2}, sample);
            
        end
        
    end

end