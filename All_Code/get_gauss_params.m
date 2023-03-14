function [A,C_1,m]=get_gauss_params(X)
C=cov(X');
[~,S,~] = svd(C);
temp=S(S~=0);
a=10^(-sum(log10(temp))/length(temp));
C_1=a*(a*C)^-1;
A=-(size(C,1)/2)*log(2*pi/a)-0.5*log(det(a*C));
m=mean(X,2);
end