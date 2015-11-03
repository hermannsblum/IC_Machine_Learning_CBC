function [ confusionMatrix ] = crossValidation( examples, attributes, labels )
%crossValidation performs a 10-fold cross validation (90% training and 
%validation, 10% testing) for the decision tree algorithm and reports the 
%average performances in terms of:
% - Confusion matrix;
% - Precision and Recall;
% - F1 measure;
% - Classification rate. 
%The function evaluate to different models for make predicition, i.e
%testTrees1 and testTrees2. T

%Define the folds
fold(1, :)  = 1:900;
fold(2, :)  = [1:800, 901:1000];
fold(3, :)  = [1:700, 801:1000];
fold(4, :)  = [1:600, 701:1000];
fold(5, :)  = [1:500, 601:1000];
fold(6, :)  = [1:400, 501:1000];
fold(7, :)  = [1:300, 401:1000];
fold(8, :)  = [1:200, 301:1000];
fold(9, :)  = [1:100, 201:1000];
fold(10, :) = 101:1000;

confusionMatrix = zeros(6);
%Cross validation
for i = 1:10
   
    %Train the classifier
    T(i, :) = train(examples(fold(i, :), :), attributes, labels(fold(i, :)));
    
    %Test the classifier
    test_examples = examples;
    test_examples(fold(i, :), :) = []; 
    predictions(:, i) = testTrees1(T(i, :),test_examples);
    
    %Compute the error
    [N, ~] = size(test_examples);
    test_labels = labels;
    test_labels(fold(i, :)) = []; 
    class_error(i) = 1/N * sum(not(predictions(:, i) == test_labels));
    
    for j=1:length(test_labels)
        confusionMatrix(test_labels(j),predictions(j,i)) = confusionMatrix(test_labels(j),predictions(j,i))+1;
    end
end

avg_class_error = mean(class_error);

end

