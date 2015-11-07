load('cleandata_students.mat')

data_size = length(y);
% perform random permutation on training data
permutation = randperm(data_size);
examples = x(permutation, :);
emotions = y(permutation);

%split a training data set
split = round(0.6 * data_size);
training_examples = examples(1:split, :);
training_emotions = emotions(1:split);
test_examples = examples(split+1:data_size, :);
test_emotions = emotions(split+1:data_size);

% train the trees
[M, T] = mother_train(training_examples, 1:size(training_examples, 2), training_emotions);


% test in testset
prediction = mothers_decision(M, T, test_examples);

confusion = confusionmat(test_emotions, prediction);
fprintf('Confusion Matrix \n');
disp(confusion);

accuracy = sum(diag(confusion))/sum(sum(confusion));

fprintf('Performace is %0.5f \n', accuracy);

