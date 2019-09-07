%
%  example of a 2D Regression
%
%load trn;
%load tr_tr;
%load tin;
%-----------Data creation------------------------------
 output=zeros(1)
for i=0:122
    
    q=346+i;
    z=347+i;
   k=tr_tr(1:q,1);
        l=trn(1:q,1);
tin=trn(z:z,1);
        for a=2:16
           
       ab=trn(1:q,a);
  l=cat(2,l,ab);
  cd=trn(z:z,a);
  tin=cat(2,tin,cd);
        end
        
        % x1=2*randn(N,1);
% x2=2*randn(N,1);
% y=exp(-(x1.^2+x2.^2)*2);
% x=[x1 x2];
%x=trn;
%y=tr_tr;

%------------------------------------------------------

C = 60; 
lambda = .010; 
epsilon = .01;
kerneloption = 01;
kernel='poly';
verbose=1;

[xsup,ysup,w,w0] = svmreg(trn,tr_tr,C,epsilon,kernel,kerneloption,lambda,verbose);

%--------------------------------------------------------
% [xtesta1,xtesta2]=meshgrid([-3:0.1:3],[-3:0.1:3]);
% [na,nb]=size(xtesta1);
% xtest1=reshape(xtesta1,1,na*nb);
% xtest2=reshape(xtesta2,1,na*nb);
% xtest=[xtest1;xtest2]';
%xtest=tin;
output1 = svmval(tin,xsup,w,w0,kernel,kerneloption);
output=cat(1,output,output1)
% ypredmat=reshape(ypred,na,nb);
% 
% %--------------------------------------------------------
% mesh(xtesta1,xtesta2,ypredmat);
% xlabel('x');
% ylabel('y');
% title('Support Vector Machine Regression');
% 

end
