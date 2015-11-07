function [ output_args ] = randomizedCrossValidation( examples, attributes, labels )
%crossValidation performs a 10-fold cross validation (90% training and 
%validation, 10% testing) for the decision tree algorithm and reports the 
%average performances in terms of:
% - Confusion matrix;
% - Precision and Recall;
% - F1 measure;
% - Classification rate. 
%The function evaluate to different models for make predicition, i.e
%testTrees1 and testTrees2. T

randomIndexes = randperm(1000);
%Define the folds
fold(1, :)  = randomIndexes(1:900);
fold(2, :)  = [randomIndexes(1:800), randomIndexes(901:1000)];
fold(3, :)  = [randomIndexes(1:700), randomIndexes(801:1000)];
fold(4, :)  = [randomIndexes(1:600), randomIndexes(701:1000)];
fold(5, :)  = [randomIndexes(1:500), randomIndexes(601:1000)];
fold(6, :)  = [randomIndexes(1:400), randomIndexes(501:1000)];
fold(7, :)  = [randomIndexes(1:300), randomIndexes(401:1000)];
fold(8, :)  = [randomIndexes(1:200), randomIndexes(301:1000)];
fold(9, :)  = [randomIndexes(1:100), randomIndexes(201:1000)];
fold(10, :) = randomIndexes(101:1000);

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
    class_error(i) = 1/N * sum(not(predictions(:, i) == test_labels))
    
    % Build the confusion matrix
    for j=1:length(test_labels)
        confusionMatrix(test_labels(j),predictions(j,i)) = confusionMatrix(test_labels(j),predictions(j,i))+1;
    end
end

avg_class_error = mean(class_error);

end

