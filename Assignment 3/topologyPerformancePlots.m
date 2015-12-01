% This script produces the optimal parameters
getOptimalParameters;

% Gradient descent: the optimal parameters' indices are inside
% multi_imin_gd
lr_index = multi_imin_gd{3};
traingd_performancesOneHidden = traingd_msErrors(:,1,lr_index);
traingd_performancesTwoHidden = traingd_msErrors(:,2,lr_index);

figure;
plot(6:45,traingd_performancesOneHidden,6:45,traingd_performancesTwoHidden);
title('Gradient descent');
xlabel('Neurons per hidden layer');
ylabel('MSE');
legend('1 hidden layer','2 hidden layers');

% Adaptive gradient descent
lr_index = multi_imin_gda{3};
lr_dec_index = multi_imin_gda{4};
lr_inc_index = multi_imin_gda{5};

traingda_performancesOneHidden = traingda_msErrors(:,1,lr_index,lr_dec_index,lr_inc_index);
traingda_performancesTwoHidden = traingda_msErrors(:,2,lr_index,lr_dec_index,lr_inc_index);

figure;
plot(6:45,traingda_performancesOneHidden,6:45,traingda_performancesTwoHidden);
title('Adaptive gradient descent');
xlabel('Neurons per hidden layer');
ylabel('MSE');
legend('1 hidden layer','2 hidden layers');

% With momentum
lr_index = multi_imin_gdm{3};
mc_index = multi_imin_gdm{4};
traingdm_performancesOneHidden = traingdm_msErrors(:,1,lr_index,mc_index);
traingdm_performancesTwoHidden = traingdm_msErrors(:,2,lr_index,mc_index);

figure;
plot(6:45,traingdm_performancesOneHidden,6:45,traingdm_performancesTwoHidden);
title('Gradient descent with momentum');
xlabel('Neurons per hidden layer');
ylabel('MSE');
legend('1 hidden layer','2 hidden layers');

% Resilient backpropagation
delt_inc_index = multi_imin_rp{3};
delt_dec_index = multi_imin_rp{4};
trainrp_performancesOneHidden = trainrp_msErrors(:,1,delt_inc_index,delt_dec_index);
trainrp_performancesTwoHidden = trainrp_msErrors(:,2,delt_inc_index,delt_dec_index);

figure;
plot(6:45,trainrp_performancesOneHidden,6:45,trainrp_performancesTwoHidden);
title('Resilient backpropagation');
xlabel('Neurons per hidden layer');
ylabel('MSE');
legend('1 hidden layer','2 hidden layers');