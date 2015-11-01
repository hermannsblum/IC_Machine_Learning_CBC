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
T = train(training_examples, 1:size(training_examples, 2), training_emotions);

% test in testset
prediction = decide_by_score(T, test_examples);

error_rate = sum(abs(prediction - test_emotions))/(data_size - split);

fprintf('Performace is %0.5f \n', 1-error_rate);