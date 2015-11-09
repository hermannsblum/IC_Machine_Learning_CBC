load('noisydata_students.mat')

data_size = length(y);

times = 1;

confusion = cell(times,1);
accuracy = zeros(times, 1);
precision = zeros(times, 1);
recall = zeros(times, 1);

T = train(x, 1:45, y);
% for i=1:6
%     DrawDecisionTree(T(i));
% end

[confusion, acc, prec, rec, f1] = crossValidate(x, y, 10, true);
accuracy = acc;
precision = prec;
recall = rec;


fprintf('confusion matrix \n');
disp(confusion);



fprintf('Accuracy is %0.5f \n', mean(accuracy));
for i = 1:length(precision)
    fprintf('Class %d: Prec %0.5f, Recall %0.5f, F1 %0.5f \n', i, precision(i), recall(i), f1(i));
end
%latex style
for i = 1:length(precision)
    fprintf('%d & %0.1f\\%% & %0.1f\\%% & %0.1f\\%% \\\\ \n', i, precision(i)*100, recall(i) * 100, f1(i)*100);
end

fprintf('F1 is %0.5f \n', mean(f1));
