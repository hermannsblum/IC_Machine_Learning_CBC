fprintf('Clean Data \n');
load('cleandata_students.mat');
pruning_example(x, y);
figure;
fprintf('noisy Data \n');
load('noisydata_students.mat')
pruning_example(x, y);