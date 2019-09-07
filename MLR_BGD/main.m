clear; close all; clc;

load('Houston_AQI.mat');

k1=1;
k2=1500;
inputs_train = inputs_original(k1:k2,:);
inputs_test = inputs_original((k2+1):end,:);
target_train = AQI_original(k1:k2,:);
target_test = AQI_original((k2+1):end,:);
 m=length(inputs_train);
[inputs_train,i_avg,i_sd]=meannormalise(inputs_train);
[target,t_avg,t_sd]=meannormalise(target_train);

theta=randomlyinitializeweights(size(inputs_train,2),1);

inputs_train = [ones(m, 1) inputs_train];
alpha = 0.01;

epochs=500;

[theta, J] = gradientdescent(inputs_train, target, theta, alpha, epochs);

n=length(inputs_test);
inputs_test=bsxfun(@minus,inputs_test,i_avg);
inputs_test=bsxfun(@rdivide,inputs_test,i_sd);
inputs_test = [ones(n, 1) inputs_test];

rows = size(inputs_test,1);
Observed = zeros(rows,1);
Predicted = zeros(rows,1);

pred_train=inputs_train*theta;
pred_test=inputs_test*theta;
pred_train_org=(pred_train*t_sd)+t_avg;
pred_test_org=(pred_test*t_sd)+t_avg;

Observed = target_test;
Predicted = pred_test_org;

figure(1);
plot(Observed,'-.g*','DisplayName','target_observed');hold on;plot(Predicted,':bs','DisplayName','target_predicted');hold off;
xlabel('Day');
ylabel('AQI');
legend('Observed','Predicted');
figure(2);
plotregression(Observed,Predicted,'Regression');

MAPE_test=mape(target_test, pred_test_org)*100

MAE_test=mean_abs_err(target_test, pred_test_org)

R_test=corr2(target_test, pred_test_org)

IA_test=index_of_agreement(target_test, pred_test_org)

RMSE_test=rmse(target_test, pred_test_org)








