clear; close all; clc

input_layer_size=5;
hidden_layer_size=5;
output_layer_size=1;

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

disp('Initialize Random Weights');
wih = randomlyinitializeweights(input_layer_size, hidden_layer_size);
who = randomlyinitializeweights(hidden_layer_size, output_layer_size);
wi1h = randomlyinitializeweights(input_layer_size-1, hidden_layer_size);
wi2h = randomlyinitializeweights(input_layer_size-1, hidden_layer_size);
wo1h = randomlyinitializeweights(output_layer_size-1, hidden_layer_size);
wo2h = randomlyinitializeweights(output_layer_size-1, hidden_layer_size);
disp('Random weights assigned');
    
params = [wih(:) ; who(:) ; wi1h(:) ; wi2h(:) ; wo1h(:) ; wo2h(:)];
fprintf('\n Forward Propagation \n');
    
J = costfunction(params, input_layer_size, hidden_layer_size, output_layer_size, inputs_train, target);
    
fprintf('\nTraining Neural Network... \n')
    
options = optimset('MaxIter', 500);
    
newcostfunction = @(p) costfunction(p,input_layer_size,hidden_layer_size,output_layer_size, inputs_train, target);
    
[params, cost] = fmincg(newcostfunction,params, options);
    
wih = reshape(params(1:hidden_layer_size*(input_layer_size+1)), hidden_layer_size, (input_layer_size + 1));
[m1,n1]=size(wih);
who = reshape(params((1 + (m1*n1)):(hidden_layer_size + 1 + (m1*n1))),output_layer_size, (hidden_layer_size + 1));
wi1h = reshape(params((2+hidden_layer_size+(m1*n1)):((input_layer_size*hidden_layer_size)+1+hidden_layer_size+(m1*n1))),hidden_layer_size,input_layer_size);
wi2h = reshape(params(((input_layer_size*hidden_layer_size)+2+hidden_layer_size+(m1*n1)):((2*input_layer_size*hidden_layer_size)+1+hidden_layer_size+(m1*n1))),hidden_layer_size,input_layer_size);
wo1h = reshape(params(((2*input_layer_size*hidden_layer_size)+2+hidden_layer_size+(m1*n1)):((2*input_layer_size*hidden_layer_size)+1+(2*hidden_layer_size)+(m1*n1))),hidden_layer_size,output_layer_size);
wo2h = reshape(params(((2*input_layer_size*hidden_layer_size)+2+(2*hidden_layer_size)+(m1*n1)):((2*input_layer_size*hidden_layer_size)+1+(3*hidden_layer_size)+(m1*n1))),hidden_layer_size,output_layer_size);
    
inputs_test=bsxfun(@minus,inputs_test,i_min);
inputs_test=bsxfun(@rdivide,inputs_test,(i_max-i_min));
    
pred_train = predict_out(inputs_train,wih,who,wi1h,wi2h,wo1h,wo2h,hidden_layer_size);
    
pred_test = predict_out(inputs_test,wih,who,wi1h,wi2h,wo1h,wo2h,hidden_layer_size);
    
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










