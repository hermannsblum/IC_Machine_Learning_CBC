function [ predictions ] = testTreesSimplestHypothesis( T, test_data )
%testTreesSimplestHypothsis makes prediction on the unseen data test_data using the 6 trees
%in T and returns the results in predictions. The decision method chooses
%the label which is found by the node with the lowest depth in case of a tie.

[m, ~] = size(test_data);
binary_predictions = zeros(m, 6);
depth_predictions = zeros(m, 6);
predictions=zeros(m,1);

for i = 1:m
    
    for j = 1:6
        
        c=1;                    % counter             
        tree=T(j);              % testing Tree 
        done=0;
        class=-1; 
        
        while done ~=1          % count depth and find class
            [tree,test_data(i,:), done, class] = predictionBinaryTree(tree,test_data(i,:));
            c=c+1;
        end
        
        binary_predictions(i, j) = class;           % save labels
        depth_predictions(i, j) = c;                % save depth
        
    end
    
end

for i=1:m
    if sum(binary_predictions(i,:))>1                           % more than one label found
        a=depth_predictions(i,:).*binary_predictions(i,:);  
        a(a==0)=46;
        [~,predictions(i)]=min(a);                              % choose minimum depth
    elseif sum(binary_predictions(i,:))==1
        [~, predictions(i)] = max(binary_predictions(i,:),[],2);    % only one label assigned
    else                                                        % all labels are 0
        predictions(i)=1;                                      
    end
end

end



function [tree, x, done, class ] = predictionBinaryTree( tree, x)
%predictionBinaryTree walks along the tree to find the classification of
%the istance x and calculates the depth
done=0;
class=-1;

if (isempty(tree.op))   %Check if it is a leaf
    
    class = tree.class;
    done = 1;           % counting done
    
else                    %Test the attribute
    
    x_to_test = x(tree.op);
    
    if (x_to_test == 0) %Follow the left branch
        
        tree = tree.kids{1};
        
    else %Follow the right branch
        
        tree = tree.kids{2};
        
    end
    
end

end