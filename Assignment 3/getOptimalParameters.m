load('traingd_avgmsErrors.mat');
traingd_msErrors = mserrors;
traingd_parameters = parameters;

load('traingda_avgmsErrors.mat');
traingda_msErrors = mserrors;
traingda_parameters = parameters;

load('traingdm_avgmsErrors.mat');
traingdm_msErrors = mserrors;
traingdm_parameters = parameters;

load('trainrp_avgmsErrors.mat');
trainrp_msErrors = mserrors;
trainrp_parameters = parameters;

load('cleandata_students.mat');
[xNN,yNN] = ANNdata(x,y);

[min_traingd,imin_traingd] = min(traingd_msErrors(:));
% Get the multidimensional indices of the minimum error
multi_imin_gd = cell(1,3);
[multi_imin_gd{:}] = ind2sub(size(traingd_msErrors),imin_traingd);

[min_traingda,imin_traingda] = min(traingda_msErrors(:));
% Get the multidimensional indices of the minimum error
multi_imin_gda = cell(1,5);
[multi_imin_gda{:}] = ind2sub(size(traingda_msErrors),imin_traingda);

[min_traingdm,imin_traingdm] = min(traingdm_msErrors(:));
% Get the multidimensional indices of the minimum error
multi_imin_gdm = cell(1,4);
[multi_imin_gdm{:}] = ind2sub(size(traingdm_msErrors),imin_traingdm);

[min_trainrp,imin_trainrp] = min(trainrp_msErrors(:));
% Get the multidimensional indices of the minimum error
multi_imin_rp = cell(1,4);
[multi_imin_rp{:}] = ind2sub(size(trainrp_msErrors),imin_trainrp);

[~,ibestAlgorithm] = min([min_traingd min_traingda min_traingdm min_trainrp]);

switch ibestAlgorithm
    case 1
        npl = traingd_parameters{1}(multi_imin_gd{1});
        l = traingd_parameters{2}(multi_imin_gd{2});
        lr = traingd_parameters{3}(multi_imin_gd{3});
        
        net = feedforwardnet(repmat(npl,1,l),'traingd');
        net.trainParam.lr = lr;
        [~,net] = repeatNNTraining(net,xNN,yNN,1:900,901:1001);
        save('NN.mat','net');
        
    case 2
        npl = traingda_parameters{1}(multi_imin_gda{1});
        l = traingda_parameters{2}(multi_imin_gda{2});
        lr = traingda_parameters{3}(multi_imin_gda{3});
        lr_dec = traingda_parameters{4}(multi_imin_gda{4});
        lr_inc = traingda_parameters{5}(multi_imin_gda{5});
        
        net = feedforwardnet(repmat(npl,1,l),'traingda');
        net.trainParam.lr = lr;
        net.trainParam.lr_dec = lr_dec;
        net.trainParam.lr_inc = lr_inc;
        [~,net] = repeatNNTraining(net,xNN,yNN,1:900,901:1001);
        save('NN.mat','net');        
        
    case 3
        npl = traingdm_parameters{1}(multi_imin_gdm{1});
        l = traingdm_parameters{2}(multi_imin_gdm{2});
        lr = traingdm_parameters{3}(multi_imin_gdm{3});
        mc = traingdm_parameters{4}(multi_imin_gdm{4});
        
        net = feedforwardnet(repmat(npl,1,l),'traingdm');
        net.trainParam.lr = lr;
        net.trainParam.mc = mc;
        [~,net] = repeatNNTraining(net,xNN,yNN,1:900,901:1001);
        save('NN.mat','net');
        
    case 4
        npl = trainrp_parameters{1}(multi_imin_rp{1});
        l = trainrp_parameters{2}(multi_imin_rp{2});
        delt_inc = trainrp_parameters{3}(multi_imin_rp{3});
        delt_dec = trainrp_parameters{4}(multi_imin_rp{4});
        
        net = feedforwardnet(repmat(npl,1,l),'trainrp');
        net.trainParam.delt_inc = delt_inc;
        net.trainParam.delt_dec = delt_dec;
        [~,net] = repeatNNTraining(net,xNN,yNN,1:900,901:1001);
        save('NN.mat','net');
        
        
end