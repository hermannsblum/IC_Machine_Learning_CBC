function [ predictions ] = testTreesFirstChoice( T, test_data )
%testTreesFirstChoice makes prediction on the unseen data test_data using the 6 trees
%in T and returns the results in predictions. The decision method choose
%the maximum prediction (0 or 1) and choose the first (beetween the trees whom output 1
%prediction in case of a tie.

[m, ~] = size(test_data);
binary_predictions = zeros(m, 6);

for i = 1:m

    for j = 1:6
        
        binary_predictions(i, j) = predictionBinaryTree(T(j), test_data(i, :));
   
    end

end

[~, predictions] = max(binary_predictions,[],2);

end

function [ class ] = predictionBinaryTree( tree, x )
%predictionBinaryTree walks along the tree to find the classification of
%the istance x
    
    if (isempty(tree.op))   %Check if it is a leaf
        
        class = tree.class;
        
    else                    %Test the attribute
        
        x_to_test = x(tree.op);
        
        if (x_to_test == 0) %Follow the left branch
           
           class = predictionBinaryTree(tree.kids{1}, x);
            
        else %Follow the right branch
            
           class = predictionBinaryTree(tree.kids{2}, x);
            
        end
        
    end

end