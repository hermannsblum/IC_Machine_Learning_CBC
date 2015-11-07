function [ predictions ] = testTreesRandomChoice(T,x2)
% TestTreesRandomChoice performs a random selection among the emotions of
% all the trees that return a prediction of 1, or among all the emotions if
% all the trees return 0.

[m,n] = size(x2);
binary_predictions = zeros(m,6);
predictions = zeros(m,1);

for i = 1:m
    for j = 1:6
        binary_predictions(i,j) = predictionBinaryTree(T(j),x2(i,:));
    end
    % If all the predictions are 0, pick an emotion randomly
    if(max(binary_predictions(i,:))==0)
        predictions(i) = randi(6);
    % Else pick an emotion randomly among the 1s
    else
        candidateEmotions = find(binary_predictions(i,:));
        predictions(i) = candidateEmotions(randi(length(candidateEmotions)));
    end
    
end


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