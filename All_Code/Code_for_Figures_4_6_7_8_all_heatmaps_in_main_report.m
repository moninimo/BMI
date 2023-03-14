%% CODE FOR GENERATING DATA FOR FIGURES 4,6,7,8 IN REPORT
clc;clear;close all;
load 'monkeydata_training.mat'
max_iter=50;
M_lda=7;
T_vec=320:20:560;
c=1;
acc=zeros([length(T_vec) 4 max_iter 3 2]);
mse_if_last=zeros([length(T_vec) 4 max_iter 3 2]);
mse_up_to=zeros([4 max_iter 3 2]);
for iter=1:max_iter
    iter
    tic
    [train,test]=split_data(trial,0.7);
    [avgx,avgy]=get_all_handPos(train);
    [~,~,x,y,~,in_data]=get_all_handPos(test);
    while size(avgx,2)<size(x,2)
        avgx=[avgx avgx(:,end,:)];
        avgy=[avgy avgy(:,end,:)];
    end
    avgx=avgx(:,1:size(x,2),:);avgy=avgy(:,1:size(x,2),:);
    x=x(:,320:20:end,:);y=y(:,320:20:end,:);
    in_data=in_data(:,320:20:end,:);
    x1=[];y1=[];in_data1=[];
    for k=1:8
        x1=[x1;x(:,:,k)];
        y1=[y1;y(:,:,k)];
        in_data1=[in_data1;in_data(:,:,k)];
    end
    avgx=avgx(:,320:20:end,:);avgy=avgy(:,320:20:end,:);
    [F_20,l,t_20,mF_20]=organize_data1(train,20,T_vec(end));
    [F_test_20,l_test,t_test_20]=organize_data1(test,20,T_vec(end));
    [F_40,l,t_40,mF_40]=organize_data1(train,40,T_vec(end));
    [F_test_40,l_test,t_test_40]=organize_data1(test,40,T_vec(end));
    [F_80,l,t_80,mF_80]=organize_data1(train,80,T_vec(end));
    [F_test_80,l_test,t_test_80]=organize_data1(test,80,T_vec(end));
    for dt=[20 40 80]
        if dt==20
            F=F_20;
            F_test=F_test_20;
            t=t_20;
            mF=mF_20;
            t_test=t_test_20;
            M_pca_kNN=35;M_pca_SVM=85;M_pca_bayes=65;s2=0.1;
            dt_index=1;
        elseif dt==40
            F=F_40;
            F_test=F_test_40;
            t=t_40;
            t_test=t_test_40;
            mF=mF_40;
            M_pca_kNN=260;M_pca_SVM=190;M_pca_bayes=69;s2=0.1;
            dt_index=2;
        elseif dt==80
            F=F_80;
            F_test=F_test_80;
            t_test=t_test_80;
            t=t_80;
            mF=mF_80;
            M_pca_kNN=170;M_pca_SVM=180;M_pca_bayes=31;s2=0.07;
            dt_index=3;
        end
        for no_pca_lda=[0 1]
            for ind_T=1:length(T_vec)
                T=T_vec(ind_T);
                X=F(t<=T,:);Xt=F_test(t_test<=T,:);
                
                if no_pca_lda==0
                    [N,mx,Wpca,L]=eigenmodel(X,size(X,2));
                    [sb,sw]=make_sb_sw(X,l);
                    M_pca=M_pca_kNN;
                    Wopt=make_Wopt(Wpca,M_pca,M_lda,sb,sw);
                    W=Wopt'*(X-mx);
                    Wt=Wopt'*(Xt-mx);
                else
                    W=X;Wt=Xt;
                end
                Wmean=zeros([size(Wt,1) 8]);
                for k=1:8; Wmean(:,k)=mean(W(:,l==k),2); end
                [pred]=do_kNN_fast(2,Wt,Wmean,1:8,1);
                [er_if_last, er_at_time]=errors_at_different_times(pred,ind_T,x1,y1,avgx,avgy,in_data1);
                mse_if_last(ind_T,1,iter,dt_index,no_pca_lda+1)=mse_up_to(1,iter,dt_index,no_pca_lda+1)+er_if_last;
                mse_up_to(1,iter,dt_index,no_pca_lda+1)=mse_up_to(1,iter,dt_index,no_pca_lda+1)+er_at_time;
                acc(ind_T,1,iter,dt_index,no_pca_lda+1)=mean(pred==l_test,2);
                pred_kNN=pred;
                if no_pca_lda==0
                    M_pca=M_pca_SVM;
                    Wopt=make_Wopt(Wpca,M_pca,M_lda,sb,sw);
                    W=Wopt'*(X-mx);
                    Wt=Wopt'*(Xt-mx);
                else
                    W=X;Wt=Xt;
                end
                Wmean=zeros([size(Wt,1) 8]);
                for k=1:8; Wmean(:,k)=mean(W(:,l==k),2); end
                [pred]=do_SVM(Wt,Wmean,s2,c);
                [er_if_last, er_at_time]=errors_at_different_times(pred,ind_T,x1,y1,avgx,avgy,in_data1);
                mse_if_last(ind_T,2,iter,dt_index,no_pca_lda+1)=mse_up_to(2,iter,dt_index,no_pca_lda+1)+er_if_last;
                mse_up_to(2,iter,dt_index,no_pca_lda+1)=mse_up_to(2,iter,dt_index,no_pca_lda+1)+er_at_time;
                acc(ind_T,2,iter,dt_index,no_pca_lda+1)=mean(pred==l_test,2);
                pred_SVM=pred;
                
                if no_pca_lda==0
                    M_pca=M_pca_bayes;
                    Wopt=make_Wopt(Wpca,M_pca,M_lda,sb,sw);
                    W=Wopt'*(X-mx);
                    Wt=Wopt'*(Xt-mx);
                    p=[];
                    for k=1:8
                        W_c=W(:,l==k);
                        [A,C_1,m]=get_gauss_params(W_c);
                        Y=Wt-m;
                        temp1=diag(Y'*C_1*Y);
                        p1=exp(A-0.5*temp1);
                        p=[p;p1'];
                    end
                    [~,pred]=max(p);
                    [er_if_last, er_at_time]=errors_at_different_times(pred,ind_T,x1,y1,avgx,avgy,in_data1);
                    mse_if_last(ind_T,3,iter,dt_index,no_pca_lda+1)=mse_up_to(3,iter,dt_index,no_pca_lda+1)+er_if_last;
                    mse_up_to(3,iter,dt_index,no_pca_lda+1)=mse_up_to(3,iter,dt_index,no_pca_lda+1)+er_at_time;
                    acc(ind_T,3,iter,dt_index,no_pca_lda+1)=mean(pred==l_test,2);
                    pred_bayes=pred;
                    
                    [pred,freq]=mode([pred_kNN;pred_SVM;pred_bayes]);
                    pred(freq==1)=pred_SVM(freq==1);
                    [er_if_last, er_at_time]=errors_at_different_times(pred,ind_T,x1,y1,avgx,avgy,in_data1);
                    mse_if_last(ind_T,4,iter,dt_index,no_pca_lda+1)=mse_up_to(4,iter,dt_index,no_pca_lda+1)+er_if_last;
                    mse_up_to(4,iter,dt_index,no_pca_lda+1)=mse_up_to(4,iter,dt_index,no_pca_lda+1)+er_at_time;
                    acc(ind_T,4,iter,dt_index,no_pca_lda+1)=mean(pred==l_test,2);
                end
            end
            
            mse_if_last(:,:,iter,dt_index,no_pca_lda+1)=mse_if_last(:,:,iter,dt_index,no_pca_lda+1)/sum(sum(in_data1));
        end
    end
    toc
end
acc_mean=mean(acc,3);
rmse_mean=mean(sqrt(mse_if_last),3);
acc_std=std(acc,0,3);
rmse_std=std(sqrt(mse_if_last),[],3);
%%
F4_acc_mean=[acc_mean(:,1,1,1,2) acc_mean(:,1,1,1,1)...
    acc_mean(:,1,1,2,2) acc_mean(:,1,1,2,1)...
    acc_mean(:,1,1,3,2) acc_mean(:,1,1,3,1)];
F4_rmse_mean=[rmse_mean(:,1,1,1,2) rmse_mean(:,1,1,1,1)...
    rmse_mean(:,1,1,2,2) rmse_mean(:,1,1,2,1)...
    rmse_mean(:,1,1,3,2) rmse_mean(:,1,1,3,1)];
F4_rmse_std=[rmse_std(:,1,1,1,2) rmse_std(:,1,1,1,1)...
    rmse_std(:,1,1,2,2) rmse_std(:,1,1,2,1)...
    rmse_std(:,1,1,3,2) rmse_std(:,1,1,3,1)];
figure(1)
h=heatmap(F4_acc_mean)
title("figure_4_(kNN)_accuracy_mean")
h.YData=(320:20:560);
figure(2)
subplot(1,2,1)
h=heatmap(F4_rmse_mean)
title("figure_4_(kNN)_rmse_mean")
h.YData=(320:20:560);
subplot(1,2,2)
h=heatmap(F4_rmse_std)
title("figure_4_(kNN)_rmse_std_not_in_report")
h.YData=(320:20:560);

F6_acc_mean=[acc_mean(:,2,1,1,2) acc_mean(:,2,1,1,1)...
    acc_mean(:,2,1,2,2) acc_mean(:,2,1,2,1)...
    acc_mean(:,2,1,3,2) acc_mean(:,2,1,3,1)];
F6_rmse_mean=[rmse_mean(:,2,1,1,2) rmse_mean(:,2,1,1,1)...
    rmse_mean(:,2,1,2,2) rmse_mean(:,2,1,2,1)...
    rmse_mean(:,2,1,3,2) rmse_mean(:,2,1,3,1)];
F6_rmse_std=[rmse_std(:,2,1,1,2) rmse_std(:,2,1,1,1)...
    rmse_std(:,2,1,2,2) rmse_std(:,2,1,2,1)...
    rmse_std(:,2,1,3,2) rmse_std(:,2,1,3,1)];
figure(3)
h=heatmap(F6_acc_mean)
title("figure_6_(SVM)_acc_mean")
h.YData=(320:20:560);
figure(4)
subplot(1,2,1)
h=heatmap(F6_rmse_mean)
title("figure_6_(SVM)_rmse_mean")
h.YData=(320:20:560);
subplot(1,2,2)
h=heatmap(F6_rmse_std)
title("figure_6_(SVM)_rmse_std_not_in_report")
h.YData=(320:20:560);

F7_acc_mean=[acc_mean(:,3,1,1,1) acc_mean(:,3,1,2,1) acc_mean(:,3,1,3,1)];
F7_rmse_mean=[rmse_mean(:,3,1,1,1) rmse_mean(:,3,1,2,1) rmse_mean(:,3,1,3,1)];
F7_rmse_std=[rmse_std(:,3,1,1,1) rmse_std(:,3,1,2,1) rmse_std(:,3,1,3,1)];

figure(5)
h=heatmap(F7_acc_mean)
title("figure_7_(Bayes)_acc_mean")
h.YData=(320:20:560);
figure(6)
subplot(1,2,1)
h=heatmap(F7_rmse_mean)
title("figure_7_(Bayes)_rmase_mean")
h.YData=(320:20:560);
subplot(1,2,2)
h=heatmap(F7_rmse_std)
h.YData=(320:20:560);
title("figure_7_(Bayes)_rmase_std_not_in_report")

F8_acc_mean=[acc_mean(:,1,1,3,1) acc_mean(:,2,1,3,1) acc_mean(:,3,1,3,1) acc_mean(:,4,1,3,1)];
F8_rmse_mean=[rmse_mean(:,1,1,3,1) rmse_mean(:,2,1,3,1) rmse_mean(:,3,1,3,1) rmse_mean(:,4,1,3,1)];
F8_rmse_std=[rmse_std(:,1,1,3,1) rmse_std(:,2,1,3,1) rmse_std(:,3,1,3,1) rmse_std(:,4,1,3,1)];

figure(7)
h=heatmap(F8_acc_mean)
h.YData=(320:20:560);
title("figure_8_(Maj_Vote)_acc_mean")
figure(8)
subplot(1,2,1)
h=heatmap(F8_rmse_mean)
h.YData=(320:20:560);
title("figure_8_(Maj_Vote)_rmse_mean")
subplot(1,2,2)
h=heatmap(F8_rmse_std)
h.YData=(320:20:560);
title("figure_8_(Maj_Vote)_rmse_std_not_in_report")


function [er_if_last, er_at_time]=errors_at_different_times(pred,ind_T,x1,y1,avgx,avgy,in_data1)
er_at_time=0;
er_if_last=0;
for tr=1:length(pred)
    er_at_time=er_at_time+...
        (x1(tr,ind_T)-avgx(1,ind_T,pred(tr)))^2+(y1(tr,ind_T)-avgy(1,ind_T,pred(tr)))^2;
    er_if_last=er_if_last+sum(...
        ((x1(tr,ind_T:end)-avgx(1,ind_T:end,pred(tr))).^2+(y1(tr,ind_T:end)-avgy(1,ind_T:end,pred(tr))).^2).*in_data1(tr,ind_T:end));
end
end