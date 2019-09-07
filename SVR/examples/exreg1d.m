%
%  1D Regression example
%
close all
% clear all
tic
%-------------------creating and plotting data----------------
n=50;
% x=linspace(0,3,n)';
% xx=linspace(0,3,n)';
x=trn;
xx=tin;
xi=x;
% yi=cos(exp(x));
yi=tr_tr;
% y=cos(exp(xx));
figure(1)
hold on;
% plot(xx,y,'g',xx,y,'b+');
%----------------------Learning Parameters -------------------
C = 6.35;  lambda = 1.3; 
epsilon = 0016;
kerneloption = .01;
kernel='gaussian';
verbose=1;
[xsup,ysup,w,w0] = svmreg(xi,yi,C,epsilon,kernel,kerneloption,lambda,verbose);
%---------------------Processing and plotting regression function--------

rx = svmval(xx,xsup,w,w0,kernel,kerneloption);
h = plot(xx,rx,'b'); 
set(h,'LineWidth',2);
h = plot(xsup,ysup,'or'); 
set(h,'LineWidth',2);
plot(xx,rx+epsilon,'b--'); 
plot(xx,rx-epsilon,'b--'); hold off;
title('Support Vector Machine Regression');
xlabel('x');ylabel('y');
hold off

