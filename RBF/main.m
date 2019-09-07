clear; close all; clc;

load('Houston_AQI.mat')

k1 = 1
k2 = 1500
rows = size(inputs_original,1);
Observed = zeros(rows,1);
Predicted = zeros(rows,1);
inputs_test = inputs_original(k1:k2,:);
inputs_train = inputs_original((k2+1):end,:);
target_test = AQI_original(k1:k2,:);
Observed(k1:k2) = target_test;
target_train = AQI_original((k2+1):end,:);

[inputs_train,i_max,i_min]=maxminnormalise(inputs_train);
[target,t_max,t_min]=maxminnormalise(target_train);

rbfneurons=10; 

% sigma=100; beta = 1 / (2*sigma^2);

disp('Training RBFN Network');

[centers, betas, theta,J] = train(inputs_train, target, rbfneurons);

% samples_test=100;
% n2=randperm(m,samples_test);
% inputs_test=x(n2,:);
% target_test=y(n2);
inputs_test=bsxfun(@minus,inputs_test,i_min);
inputs_test=bsxfun(@rdivide,inputs_test,(i_max-i_min));

xtest=inputs_train; pred_train=zeros(length(xtest),1); 

for i=1:length(xtest)
    
    pred_train(i)=evaluate(centers,betas,theta,xtest(i,:));
    
end

xtest=inputs_test; pred_test=zeros(length(xtest),1); 

for i=1:length(xtest)
    
    pred_test(i)=evaluate(centers,betas,theta,xtest(i,:));
    
end

pred_train_org=(pred_train*(t_max-t_min))+t_min;

pred_test_org=(pred_test*(t_max-t_min))+t_min;

Predicted(k1:k2) = pred_test_org;

figure(1);
plot(Observed,'-.g*','DisplayName','target_observed');hold on;plot(Predicted,':bs','DisplayName','target_predicted');hold off;
xlabel('Day');
ylabel('PM_{{2.5}} Concentration (ug/m^3)');
legend('Observed','Predicted');
figure(2);
plotregression(Observed,Predicted,'Regression');

MAPE_test=mape(Observed, Predicted)*100

MAE_test=mean_abs_err(Observed, Predicted)

R_test=corr2(Observed, Predicted)

IA_test=index_of_agreement(Observed, Predicted)

RMSE_test=rmse(Observed, Predicted)





