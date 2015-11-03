function [predictions] = mothers_decision(mother, trees, testset)

[m,n] = size(testset);
predictions = zeros(m,1);
for i = 1:m
    tree_predictions = zeros(6);
    for tree = 1:6
        tree_predictions(tree) = simple_prediction(trees(tree), testset(i, :));
    end
    predictions(i) = simple_prediction(mother, tree_predictions);
end