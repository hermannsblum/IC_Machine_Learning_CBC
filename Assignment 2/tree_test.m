load('cleandata_students.mat')

data_size = length(y);

times = 1;

confusion = cell(times,1);
accuracy = zeros(times, 1);
precision = zeros(times, 1);
recall = zeros(times, 1);

for i = 1:times
    % perform random permutation on training data
    permutation = randperm(data_size);
    examples = x(permutation, :);
    emotions = y(permutation);

    [confusion{i}, acc, prec, rec] = crossValidate( ...
        examples, emotions, 10, false);
    accuracy(i) = mean(acc);
    precision(i) = mean(prec);
    recall(i) = mean(rec);
end
% find mean confusion matrix
sum = zeros(6, 6);
for i = 1:times;
    sum = sum + confusion{i};
end

fprintf('confusion matrix \n');
disp(sum / times);

fprintf('Accuracy is %0.5f \n', mean(accuracy));
fprintf('Precision is %0.5f \n', mean(precision));
fprintf('Recall is %0.5f \n', mean(recall));