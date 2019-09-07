clear; close all; clc

input_layer_size=5;
hidden_layer_size=5;
output_layer_size=1;

load('Houston_AQI.mat')

k1=1;
k2=1500;
rows = size(inputs_original,1);
Observed = zeros(rows,1);
Predicted = zeros(rows,1);
inputs_test = inputs_original(k1:k2,:);
inputs_train = inputs_original((k2+1):end,:);
target_test = AQI_original(k1:k2,:);
target_train = AQI_original((k2+1):end,:);
Observed(k1:k2) = target_test;

    
inputs_train,i_max,i_min]=maxminnormalise(inputs_train);
[target,t_max,t_min]=maxminnormalise(target_train);
    
theta1 = randomlyinitializeweights(input_layer_size, hidden_layer_size);
theta2 = randomlyinitializeweights(hidden_layer_size, output_layer_size);
theta3 = randomlyinitializeweights(input_layer_size, output_layer_size);
    
params = [theta1(:) ; theta2(:); theta3(:)];
    
disp('\n Forward Propagation \n');
    
lambda= 0.01;
    
J = costfunction(params, input_layer_size, hidden_layer_size, output_layer_size, inputs_train, target, lambda);
    
disp('\nTraining Neural Network... \n')
    
options = optimset('MaxIter', 500);
    
newcostfunction = @(p) costfunction(p, ...
        input_layer_size, ...
        hidden_layer_size, ...
        output_layer_size, inputs_train, target, lambda);
[params, cost] = fmincg(newcostfunction, params, options);
    
theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
[m1,n1]=size(theta1);
theta2 = reshape(params((1 + (m1*n1)):(hidden_layer_size + 1 + (m1*n1))),output_layer_size, (hidden_layer_size + 1));
    
theta3 = reshape(params((1 + (1+hidden_layer_size+(m1*n1))):(input_layer_size+(1 + (1+hidden_layer_size+(m1*n1))))),output_layer_size,(input_layer_size+1));
    
    
inputs_test=bsxfun(@minus,inputs_test,i_min);
inputs_test=bsxfun(@rdivide,inputs_test,(i_max-i_min));
    
pred_train = predict_output(theta1, theta2, theta3, inputs_train);
    
pred_test = predict_output(theta1, theta2, theta3, inputs_test);
    
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





