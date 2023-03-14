%% Plot for PCR with split 0.5, 0.6, 0.7, 0.8, 0.9
%% for Higher resosultion in r (which is figure 9 in the report), set sp=0.7!
clc;clear;close all;
load monkeydata_training.mat;
sp=0.8;
max_iter=10;

if sp==0.5
    r_vec=9:10:49;
elseif sp==0.6
    r_vec=19:10:59;
elseif sp==0.7
    r_vec=2:69;
elseif sp==0.8
    r_vec=19:20:79;
elseif sp==0.9
    r_vec=9:20:89;
end
t_tot=0;
for iter=1:max_iter
    ix = randperm(length(trial));
    trainingData = trial(ix(1:sp*100),:);
    testData = trial(ix((100*sp+1):end),:);
    for rind=1:length(r_vec)
        tic
        r=r_vec(rind);
        T_end = 560;
        dt=20;
        T = (320:dt:T_end);
        [feat,l,t]=organize_data(trainingData,dt,T_end);
        [feat_test,l_test,t_test]=organize_data(testData,dt,T_end);
        [avgx,avgy,x,y,~,~]=get_all_handPos(trainingData);
        x_resampled = x(:,T,:);
        y_resampled = y(:,T,:);
        
        %create a regression model for each angle
        angle =1:8;
        
        for angle_index =1:length(angle)
            for time_index = 1:length(T)
                hpX = x_resampled(:,time_index,angle_index);
                hpY = y_resampled(:,time_index,angle_index);
                mhpX=mean(hpX);
                mhpY=mean(hpY);
                hpX=hpX-mhpX;
                hpY=hpY-mhpY;
                X = feat(t<=T(time_index),l==angle_index);
                Xt = feat_test(t_test<=T(time_index),l_test==angle_index);
                [N,mx,U,L]=eigenmodel(X,r);%U=V in wiki
                V=U(:,1:r);
                W=V'*(X-mx);%W^T in wiki
                A=V*(W*W')^(-1)*W;
                for n=1:size(Xt,2)
                    pX(n,time_index,angle_index)=(Xt(:,n)-mx)'*A*hpX+mhpX;
                    pY(n,time_index,angle_index)=(Xt(:,n)-mx)'*A*hpY+mhpY;
                end
            end
        end
        [~,~,xt,yt,~,in_data]=get_all_handPos(testData);
        xt = xt(:,320:20:end,:);
        yt = yt(:,320:20:end,:);
        avgx=avgx(:,320:20:end,:);
        avgy=avgy(:,320:20:end,:);
        id = in_data(:,320:20:end,:);
        er=0;
        er1=0;
        for i=1:size(xt,1)
            for j=1:size(xt,2)
                for k=1:size(xt,3)
                    temp=(xt(i,j,k)-pX(i,min(j,size(pX,2)),k))^2+(yt(i,j,k)-pY(i,min(j,size(pY,2)),k))^2;
                    temp1=(xt(i,j,k)-avgx(1,min(j,size(avgx,2)),k))^2+(yt(i,j,k)-avgy(1,min(j,size(avgy,2)),k))^2;
                    er=er+temp*id(i,j,k);
                    er1=er1+temp1*id(i,j,k);
                end
            end
        end
        rmse(rind,iter)=sqrt(er/sum(sum(sum(id))));
        rmse_avg(rind,iter)=sqrt(er1/sum(sum(sum(id))));
        t_elapsed(rind,iter)=toc;
        t_tot=t_tot+t_elapsed(rind,iter);
        p_complete=((iter-1)*length(r_vec)+rind)/(length(r_vec)*max_iter);
        disp(round(100*p_complete,2)+"% complete")
        disp(round((t_tot/p_complete-t_tot)/60,1)+" minutes left (estimated)")
    end
end
%%
rmse1_new = rmse';
rmse_avg1_new = rmse_avg';
mean_rmse1 = mean(rmse1_new);
mean_avg1 = mean(rmse_avg1_new);
std_rmse1 = std(rmse1_new);
std_avg1 = std(rmse_avg1_new);

mrplus = mean_rmse1 + std_rmse1;
mrminus = mean_rmse1 - std_rmse1;
msplus = mean_avg1+std_avg1;
msminus = mean_avg1-std_avg1;

% Plot mean RMSE of the split for Average trajectory and PCR
figure(1)
subplot(1,2,1);
hold off
plot(r_vec, mean_rmse1, '--o');
hold on 
plot(r_vec, mean_avg1, 'r', 'LineWidth', 1.5);
xlabel('Principal Components (r)', 'Fontsize', 18);
ylabel('mean RMSE', 'Fontsize', 18);
legend("PCR mean RMSE","Average trajectory mean RMSE")
grid on

% Plot std RMSE of the split for Average trajectory and PCR
subplot(1,2,2);
hold off
plot(r_vec, std_rmse1, 'LineWidth', 1.5);
hold on
plot(r_vec, std_avg1, 'r--', 'LineWidth', 1.5);
xlabel('Principal Components (r)', 'Fontsize', 18);
ylabel('std RMSE', 'Fontsize', 18);
legend('PCR std RMSE','Average trajectory std RMSE')
grid on