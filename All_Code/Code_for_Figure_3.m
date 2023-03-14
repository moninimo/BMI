clc;clear;close all;
load 'monkeydata_training.mat';
mean_acc=[];
mean_tim=[];
max_iter=10;
for dt=[20 40 80]
    T_vec=320;
    NN_vec=1:10:261;
    acc2=[];
    timing2=[];
    for iter=1:max_iter
        [dt iter]
        [train,test]=split_data(trial,0.7);
        [F,l,t,mF]=organize_data1(train,dt,T_vec(end));
        [F_test,l_test,t_test]=organize_data1(test,dt,T_vec(end));
        T=320;
        X=F(t<=T,:);
        Xt=F_test(t_test<=T,:);
        mX=mF(t<=T,:);
        tic
        pred=do_kNN_fast(2,Xt,X,l,NN_vec);
        timing=toc;
        acc1=100*(mean(pred==l_test,2));
        timing1=1000*timing/(length(l_test));
        tic
        pred=do_kNN_fast(2,Xt,mX,1:8,1);
        timing=toc;
        acc1=[acc1; 100*(mean(pred==l_test,2))];
        timing1=[timing1;1000*timing/(length(l_test))];
        acc2=[acc2 acc1];
        timing2=[timing2 timing1];
    end
    mean_acc=[mean_acc mean(acc2,2)];
    mean_tim=[mean_tim mean(timing2,2)];
end
hold off
plot(NN_vec',mean_acc(1:(end-1),1),':b','LineWidth',1.5)%,std(acc_no_mean,0,2));
hold on
plot(NN_vec',mean_acc(end,1)*ones(size(NN_vec')),':r','LineWidth',1.5)
plot(NN_vec',mean_acc(1:(end-1),2),'--b','LineWidth',1.5)
plot(NN_vec',mean_acc(end,2)*ones(size(NN_vec')),'--r','LineWidth',1.5)
plot(NN_vec',mean_acc(1:(end-1),3),'b','LineWidth',1.5)
plot(NN_vec',mean_acc(end,3)*ones(size(NN_vec')),'r','LineWidth',1.5)
grid on
grid minor
xlabel("Nearest Neighbours")
ylabel("Mean Accuracy (%)")
legend(["kNN with dt=20"+newline+"(average time per prediction=10ms)","Nearest Centroid with dt=20"+newline+"(average time per prediction=0.11ms)","kNN with dt=40"+newline+"(average time per prediction=7ms)","Nearest Centroid with dt=40"+newline+"(average time per prediction=0.09ms)","kNN with dt=80"+newline+"(average time per prediction=5ms)","Nearest Centroid with dt=80"+newline+"(average time per prediction=0.06ms)"],'Location','southwest','FontSize',14)
xticks(NN_vec(1:2:end))
axis square
fig=gca
fig.FontSize=12;
ylim([40 100])
xlim([1 NN_vec(end)])