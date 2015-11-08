function [ predictions ] = testTrees2( T, x, y, test_data )
%testTrees1 make prediction on the unseen data test_data using the 6 trees
%in T and returns the results in predictions. The decision method choose
%the maximum prediction (0 or 1) and choose the first (beetween the trees whom output 1
%prediction in case of a tie.

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
        binary_errors(i) = binary_errors(i) + not(binary_predictions(j, i) == y(j));
   
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

%Compute the prediciton
for i = 1:m

    value = max(binary_predictions(i, :));
    conf = 0;
  
    for j = 1:6
        
        if (binary_predictions(i, j) == value)
        
            if (confidence(j) > conf)
                
                predictions(i) = j;
                conf = confidence(j);
                
            end
            
        end
   
    end

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