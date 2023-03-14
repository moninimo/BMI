clc;clear;close all;
load 'monkeydata_training.mat'
data=trial;
[feat,l,t]=organize_data(data,20,560);
for T=320:20:560
for k=1:8
    X1=[];
    for n=1:size(data,1)
        X1=[X1 mean(data(n,k).spikes(:,1:T),2)];
    end
    X = feat(t<=T,l==k);
    [N,mx,U,L]=eigenmodel(X,size(data,1));
    clb=100*cumsum(diag(L))/sum(diag(L));
    lb=diag(L)/sum(diag(L));
    plot(100*cumsum(diag(L))/sum(diag(L)),'b')
    hold on
    grid on
    [N,mx,U,L]=eigenmodel(X1,size(data,1));
    plot(100*cumsum(diag(L))/sum(diag(L)),'r')
    lr=diag(L)/sum(diag(L));
    clr=100*cumsum(diag(L))/sum(diag(L));
    hold on
    grid on
end
end
xlabel("#Principle Components")
ylabel("%Information captured")
legend("f","f-tilde")