load('Houston_AQI.mat');

k1=1;
k2=1500;
inputs_train = inputs_original(k1:k2,:);
inputs_test = inputs_original((k2+1):end,:);
target_train = AQI_original(k1:k2,:);
target_test = AQI_original((k2+1):end,:);

[inputs_train,i_max,i_min]=maxminnormalise(inputs_train);
[target,t_max,t_min]=maxminnormalise(target_train);

% C=0.59;      
C=mean(target)+3*std(target);
epsilon=0.0;
lambda=0.5;
kerneloption = 01; 
kernel='gaussian'; 
verbose=1; 
[xsup,ysup,w,w0] = svmreg(inputs_train,target,C,epsilon,kernel,kerneloption,lambda,verbose); 
% samples_test=100;
% n2=randperm(m,samples_test);
% inputs_test=x(n2,:);
% target_test=y(n2);
inputs_test=bsxfun(@minus,inputs_test,i_min);
inputs_test=bsxfun(@rdivide,inputs_test,(i_max-i_min));
x_test=inputs_test; 
pred_train = svmval(inputs_train,xsup,w,w0,kernel,kerneloption); 
pred_test = svmval(x_test,xsup,w,w0,kernel,kerneloption); 

pred_train_org=(pred_train*(t_max-t_min))+t_min;

pred_test_org=(pred_test*(t_max-t_min))+t_min;

Observed = target_test;
Predicted = pred_test_org;

figure(1);
plot(Observed,'-.g*','DisplayName','target_observed');hold on;plot(Predicted,':bs','DisplayName','target_predicted');hold off;
xlabel('Day');
ylabel('AQI');
legend('Observed','Predicted');
figure(2);
plotregression(Observed,Predicted,'Regression');

MAPE_test=mape(target_test, pred_test_org)

MAE_test=mean_abs_err(target_test, pred_test_org)

R_test=corr2(target_test, pred_test_org)

IA_test=index_of_agreement(target_test, pred_test_org)

RMSE_test=rmse(target_test, pred_test_org)
