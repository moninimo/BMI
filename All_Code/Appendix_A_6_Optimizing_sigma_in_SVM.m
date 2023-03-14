%%CODE TO FIND OPTIMAL SIGMA GIVEN C=1 AND NO PCA-LDA
clc;clear;close all;
load 'monkeydata_training.mat'
dt=80; % VARY THIS PARAMETER TO GETR PLOTS FOR DIFFERENT DT
max_iter=2;
T_vec=320:20:560;
s2_vec=[0.02 0.05 0.1 0.2 0.5 1];
if dt==80
    s2_vec=[0.02:0.01:0.1 0.2];
end
c=1;
acc=zeros([length(T_vec) length(s2_vec)+1 max_iter]);
mse_if_last=zeros([length(T_vec) length(s2_vec)+1 max_iter]);
mse_up_to=zeros([length(s2_vec)+1 max_iter]);
timing=zeros(length(T_vec),max_iter);
for iter=1:max_iter
    iter
    tstart=tic;
    [train,test]=split_data(trial,0.7);
    [F,l,t,mF]=organize_data1(train,dt,T_vec(end));
    [F_test,l_test,t_test]=organize_data1(test,dt,T_vec(end));
    [avgx,avgy]=get_all_handPos(train);
    [~,~,x,y,~,in_data]=get_all_handPos(test);
    while size(avgx,2)<size(x,2)
        avgx=[avgx avgx(:,end,:)];
        avgy=[avgy avgy(:,end,:)];
    end
    avgx=avgx(:,1:size(x,2),:);
    avgy=avgy(:,1:size(x,2),:);
    x=x(:,320:20:end,:);
    y=y(:,320:20:end,:);
    in_data=in_data(:,320:20:end,:);
    x1=[];
    y1=[];
    in_data1=[];
    for k=1:8
        x1=[x1;x(:,:,k)];
        y1=[y1;y(:,:,k)];
        in_data1=[in_data1;in_data(:,:,k)];
    end
    avgx=avgx(:,320:20:end,:);
    avgy=avgy(:,320:20:end,:);
    
    for ind_T=1:length(T_vec)
        tic
        T=T_vec(ind_T);
        X=F(t<=T,:);
        Xt=F_test(t_test<=T,:);
        [pred]=do_kNN_fast(2,Xt,mF(t<=T,:),1:8,1);
        er_at_time=0;
        er_if_last=0;
        for tr=1:length(pred)
            er_at_time=er_at_time+...
                (x1(tr,ind_T)-avgx(1,ind_T,pred(tr)))^2+(y1(tr,ind_T)-avgy(1,ind_T,pred(tr)))^2;
            er_if_last=er_if_last+sum(...
                ((x1(tr,ind_T:end)-avgx(1,ind_T:end,pred(tr))).^2+(y1(tr,ind_T:end)-avgy(1,ind_T:end,pred(tr))).^2).*in_data1(tr,ind_T:end));
        end
        mse_if_last(ind_T,1,iter)=mse_up_to(1,iter)+er_if_last;
        mse_up_to(1,iter)=mse_up_to(1,iter)+er_at_time;
        acc(ind_T,1,iter)=mean(pred==l_test,2);
        for ind_s2=1:length(s2_vec)
            s2=s2_vec(ind_s2);
            [pred]=do_SVM(Xt,mF(t<=T,:),s2,c);
            acc(ind_T,ind_s2+1,iter)=mean(pred==l_test,2);
            er_at_time=0;
            er_if_last=0;
            for tr=1:length(pred)
                er_at_time=er_at_time+...
                    (x1(tr,ind_T)-avgx(1,ind_T,pred(tr)))^2+(y1(tr,ind_T)-avgy(1,ind_T,pred(tr)))^2;
                er_if_last=er_if_last+sum(...
                    ((x1(tr,ind_T:end)-avgx(1,ind_T:end,pred(tr))).^2+(y1(tr,ind_T:end)-avgy(1,ind_T:end,pred(tr))).^2).*in_data1(tr,ind_T:end));
            end
            mse_if_last(ind_T,ind_s2+1,iter)=mse_up_to(ind_s2+1,iter)+er_if_last;
            mse_up_to(ind_s2+1,iter)=mse_up_to(ind_s2+1,iter)+er_at_time;
        end
        timing(ind_T,iter)=toc;
    end
    mse_if_last(:,:,iter)=mse_if_last(:,:,iter)/sum(sum(in_data1));
    toc(tstart)
    
    acc_mean=mean(acc(:,:,1:iter),3);
    rmse_mean=mean(sqrt(mse_if_last(:,:,1:iter)),3);
    acc_std=std(acc(:,:,1:iter),0,3);
    rmse_std=std(sqrt(mse_if_last(:,:,1:iter)),0,3);
end

figure(1)
h=heatmap(round(100*acc_mean',2),'Colormap',bone);
h.YData=["kNN";num2str(s2_vec')];
h.XData=[num2str(T_vec')];
h.YLabel="sigma";
h.XLabel="Classification Time";
h.Title="mean ACCURACY";
h.ColorbarVisible='off';

figure(2)
subplot(1,2,1)
h=heatmap(round(rmse_mean',2),'Colormap',bone);
h.YData=["kNN";num2str(s2_vec')];
h.XData=[num2str(T_vec')];
h.YLabel="sigma";
h.XLabel="Time of last Classification";
h.Title="mean RMSE";
h.ColorbarVisible='off';
subplot(1,2,2)
h=heatmap(round(rmse_std',2),'Colormap',bone);
h.YData=["kNN";num2str(s2_vec')];
h.XData=[num2str(T_vec')];
h.YLabel="sigma";
h.XLabel="Time of last Classification";
h.Title="std(RMSE)";
h.ColorbarVisible='off';