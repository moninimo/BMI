clc;clear;close all;
load 'monkeydata_training.mat';
T=320; %this the time at which classification is performed
max_iter=1;
NN_vec=1:10:261;
for iter=1:max_iter
    iter
    tic
    [train,test]=split_data(trial,0.7);

    [X,l,~,mX]=organize_data1(train,T,T);
    [Xt,l_test,~,~]=organize_data1(test,T,T);
    pred=do_kNN_fast(2,Xt,mX,1:8,1);
    pred=[pred; do_kNN_fast(2,Xt,X,l,NN_vec)];
    if iter==1
        acc0=100*(mean(pred==l_test,2));
    else
        acc0=acc0+100*(mean(pred==l_test,2));
    end
    
    [X,l,~,mX]=organize_data1(train,20,T);
    [Xt,l_test,~,~]=organize_data1(test,20,T);
    pred=do_kNN_fast(2,Xt,mX,1:8,1);
    pred=[pred; do_kNN_fast(2,Xt,X,l,NN_vec)];
    if iter==1 
        acc20=100*(mean(pred==l_test,2));
    else
        acc20=acc20+100*(mean(pred==l_test,2));
    end
    
    [X,l,~,mX]=organize_data1(train,40,T);
    [Xt,l_test,~,~]=organize_data1(test,40,T);
    pred=do_kNN_fast(2,Xt,mX,1:8,1);
    pred=[pred; do_kNN_fast(2,Xt,X,l,NN_vec)];
    if iter==1 
        acc40=100*(mean(pred==l_test,2));
    else
        acc40=acc40+100*(mean(pred==l_test,2));
    end
    
    [X,l,~,mX]=organize_data1(train,80,T);
    [Xt,l_test,~,~]=organize_data1(test,80,T);
    pred=do_kNN_fast(2,Xt,mX,1:8,1);
    pred=[pred; do_kNN_fast(2,Xt,X,l,NN_vec)];
    if iter==1 
        acc80=100*(mean(pred==l_test,2));
    else
        acc80=acc80+100*(mean(pred==l_test,2));
    end
    toc
end
acc0=acc0'/max_iter;
acc20=acc20'/max_iter;
acc40=acc40'/max_iter;
acc80=acc80'/max_iter;

hold off
plot(NN_vec,acc0(2:end),'r');
hold on;
plot(NN_vec,acc20(2:end),'g')
plot(NN_vec,acc40(2:end),'b')
plot(NN_vec,acc80(2:end),'k')
plot(NN_vec,acc0(1)*ones(size(NN_vec)),'--r')
plot(NN_vec,acc20(1)*ones(size(NN_vec)),'--g')
plot(NN_vec,acc40(1)*ones(size(NN_vec)),'--b')
plot(NN_vec,acc80(1)*ones(size(NN_vec)),'--k')
legend("f-tilde-kNN","f-dt=20-kNN","f-dt=40-kNN","f-dt=80-kNN",...
    "f-tilde-Nearest Centroid","f-dt=20-Nearest Centroid",...
    "f-dt=40-Nearest Centroid","f-dt=80-Nearest Centroid",'Location','southeast')
grid on
xlabel('#Nearest Neighbors')
ylabel('mean accuracy (%)')
fig=gca;
fig.FontSize=14;
xlim([1 NN_vec(end)])