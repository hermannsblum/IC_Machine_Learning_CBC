function [ predictions ] = testTreesTreeConfidence( T, x, y, test_data )

[m, n] = size(test_data);
binary_predictions = zeros(m, 6);
binary_errors = zeros(1, 6);
confidence = zeros(1, 6);
predictions = zeros(m, 1);

[m_train, n_train] = size(x);

%Evaluate the error in the training set
for i = 1:6

    for j = 1:m_train
        
        binary_predictions(j, i) = predictionBinaryTree(T(i), x(j, :));
        binary_errors(i) = binary_errors(i) + (not(binary_predictions(j, i)==1 && y(j)==i)...
            && not(binary_predictions(j, i)==0 && y(j)~=i));
   
    end
    
    binary_errors(i) = 1/m_train * binary_errors(i);
    confidence(i) = 1 - binary_errors(i);

end

binary_predictions = zeros(m, 6);

%Compute the predictions of each tree
for i = 1:6

    for j = 1:m
        
        binary_predictions(j, i) = predictionBinaryTree(T(i), test_data(j, :));
   
    end

end

%Compute the prediction
for i = 1:m
    candidateClasses = find(binary_predictions(i,:)==1);
    if(isempty(candidateClasses))
        % If all the trees return 0, pick the prediction with minimum
        % confidence. This prediction, indeed, is the most error-prone and
        % most likely its correct value should be 1 (only according to our
        % confidence heuristic, there is no mathematical justification
        % here)
        [~, predictions(i)] = min(confidence);
    else
        % Take the prediction of the tree with most confidence among the
        % ones returning 1
        [~, imaxConfidence] = max(confidence(candidateClasses));
        predictions(i) = candidateClasses(imaxConfidence);
    end
%     value = max(binary_predictions(i, :));
%     conf = 0;
%   
%     for j = 1:6
%         
%         if (binary_predictions(i, j) == value)
%         
%             if (confidence(j) > conf)
%                 
%                 predictions(i) = j;
%                 conf = confidence(j);
%                 
%             end
%             
%         end
%    
%     end

end

end



function [ class ] = predictionBinaryTree( tree, x )
%predictionBinaryTree walks along the tree to find the classification of
%the istance x
    
    if (isempty(tree.op))   %Check if is a leaf
        
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