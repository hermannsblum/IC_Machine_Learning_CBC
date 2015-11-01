function [ predictions ] = decide_by_score(trees, testset)
% decide_by_score performs a selection based on the significance and
% probability of error given by the tree leafs.

[m,n] = size(testset);
binary_predictions = zeros(m,6);
predictions = zeros(m,1);

for i = 1:m
    
    % test in all trees, find the one with the best score
    best_score = 0;
    best_pred = NaN;
    for j = 1:6
        [pred, score] = prediction_with_score(trees(j),testset(i,:));
        if pred == 1
            % the tree recognises this item as his emotion
            if score > best_score
                best_pred = j;
                best_score = score;
            end
        end
    end
    % If all the predictions are 0, pick an emotion randomly
    if(isnan(best_pred))
        predictions(i) = randi(6);
    % Else pick the one with the highest score
    else
        predictions(i) = best_pred;
    end
    
end


end

function [ class, score ] = prediction_with_score( tree, x )
%prediction_with_score walks along the tree to find the classification of
%the istance x and returns in addition a score calculated from the
%significance and probability of error of the leaf
    
    if (isempty(tree.op))   %Check if it is a leaf
        
        class = tree.class;
        score = tree.significance * (1 - tree.p_error);
        
    else                    %Test the attribute
        
        x_to_test = x(tree.op);
        
        if (x_to_test == 0) %Follow the left branch
           
           [class, score] = prediction_with_score(tree.kids{1}, x);
            
        else %Follow the right branch
            
           [class, score] = prediction_with_score(tree.kids{2}, x);
            
        end
        
    end

end
