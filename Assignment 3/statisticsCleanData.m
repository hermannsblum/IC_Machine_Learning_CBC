accuracyFolds = zeros(10,1);
precisionFolds = zeros(10,6);
recallFolds = zeros(10,6);
f1Folds = zeros(10,6);
confusionMatrix = zeros(6);

for i=1:10
    load(['confMatrixFold' num2str(i) '.mat']);
    accuracyFolds(i) = sum(diag(confMatrixFold))/sum(sum(confMatrixFold));
    precisionFolds(i,:) = diag(confMatrixFold)'./sum(confMatrixFold,1);
    recallFolds(i,:) = diag(confMatrixFold)'./sum(confMatrixFold,2)';
    for j=1:6
        if precisionFolds(i,j)+recallFolds(i,j)>0
            f1Folds(i,j) = 2*precisionFolds(i,j)*recallFolds(i,j)/(precisionFolds(i,j)+recallFolds(i,j));
        end
    end
    confusionMatrix = confusionMatrix + confMatrixFold;
end
accuracy = mean(accuracyFolds);
precision = mean(precisionFolds,1);
recall = mean(recallFolds,1);
f1 = mean(f1Folds,1);