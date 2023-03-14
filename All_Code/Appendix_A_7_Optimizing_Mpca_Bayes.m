clc;clear;close all;
load 'monkeydata_training.mat'
dt=80;
max_iter=2;
M_lda=7;
T_vec=320:20:560;
M_pca_vec=9:2:69;
acc=zeros([length(T_vec) length(M_pca_vec) max_iter]);
mse_if_last=zeros([length(T_vec) length(M_pca_vec) max_iter]);
mse_up_to=zeros([length(M_pca_vec) max_iter]);
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
        [N,mx,Wpca,L]=eigenmodel(X,size(X,2));
        [sb,sw]=make_sb_sw(X,l);
        for ind_Mpca=1:length(M_pca_vec)
            M_pca=M_pca_vec(ind_Mpca);
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
            acc(ind_T,ind_Mpca,iter)=mean(pred==l_test,2);
            er_at_time=0;
            er_if_last=0;
            for tr=1:length(pred)
                er_at_time=er_at_time+...
                    (x1(tr,ind_T)-avgx(1,ind_T,pred(tr)))^2+(y1(tr,ind_T)-avgy(1,ind_T,pred(tr)))^2;
                er_if_last=er_if_last+sum(...
                    ((x1(tr,ind_T:end)-avgx(1,ind_T:end,pred(tr))).^2+(y1(tr,ind_T:end)-avgy(1,ind_T:end,pred(tr))).^2).*in_data1(tr,ind_T:end));
            end
            mse_if_last(ind_T,ind_Mpca,iter)=mse_up_to(ind_Mpca,iter)+er_if_last;
            mse_up_to(ind_Mpca,iter)=mse_up_to(ind_Mpca,iter)+er_at_time;
        end
        timing(ind_T,iter)=toc;
    end
    mse_if_last(:,:,iter)=mse_if_last(:,:,iter)/sum(sum(in_data1));
    toc(tstart)
end

acc_mean=mean(acc,3);
rmse_mean=mean(sqrt(mse_if_last),3);
acc_std=std(acc,0,3);
rmse_std=std(sqrt(mse_if_last),0,3);

figure(1)
h=heatmap(round(100*acc_mean',2),'Colormap',bone);
h.YData=[num2str(M_pca_vec')];
h.XData=[num2str(T_vec')];
h.YLabel="Mpca";
h.XLabel="Classification Time";
h.Title="mean ACCURACY";

figure(2)
subplot(1,2,1)
h=heatmap(round(rmse_mean',2),'Colormap',bone);
h.YData=[num2str(M_pca_vec')];
h.XData=[num2str(T_vec')];
h.YLabel="Mpca";
h.XLabel="Time of last Classification";
h.Title="mean RMSE";
h.ColorbarVisible='off';
subplot(1,2,2)
h=heatmap(round(rmse_std',2),'Colormap',bone);
h.YData=[num2str(M_pca_vec')];
h.XData=[num2str(T_vec')];
h.YLabel="Mpca";
h.XLabel="Time of last Classification";
h.Title="std(RMSE)";
h.ColorbarVisible='off';
