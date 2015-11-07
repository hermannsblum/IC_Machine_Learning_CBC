function [ predictions ] = decide_by_score(trees, testset)
% decide_by_score performs a classification based on the fraction of
% correctly classified training examples given by the tree leafs

[m,n] = size(testset);
predictions = zeros(m,1);

for i = 1:m
    % test in all trees, find the one with the best score
    best_score = 0;
    predictions(i) = NaN;
    for t = 1:6
        [pred, score] = prediction_with_score(trees(t),testset(i,:));
        if pred == 1
            % the tree recognises this item as his class
            if score > best_score
                predictions(i) = t;
                best_score = score;
            end
        end
    end
    % If all the predictions are 0, pick a random class
    if(isnan(predictions(i)))
        predictions(i) = randi(6);
    end
end

end

function [ class, score ] = prediction_with_score( tree, x )
%prediction_with_score walks along the tree to find the classification of
%the istance x and returns in addition a score calculated from the 
%fraction of correctly classified training examples given by the tree leafs
    
    if (isempty(tree.op))   %Check if it is a leaf
        
        class = tree.class;
        score = tree.score;
        
    else                    %Test the attribute
        
        x_to_test = x(tree.op);
        
        if (x_to_test == 0) %Follow the left branch
           
           [class, score] = prediction_with_score(tree.kids{1}, x);
            
        else %Follow the right branch
            
           [class, score] = prediction_with_score(tree.kids{2}, x);
            
        end
        
    end

end
