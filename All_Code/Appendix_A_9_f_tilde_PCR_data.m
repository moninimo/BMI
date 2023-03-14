clc;clear;close all;
load monkeydata_training.mat;
r_vec=2:1:69;
max_iter=20;
t_tot=0;
for iter=1:max_iter
    ix = randperm(length(trial));
    trainingData = trial(ix(1:70),:);
    testData = trial(ix(71:end),:);
    for rind=1:length(r_vec)
        tic
        r=r_vec(rind);
        T_end = 560;
        dt=20;
        T = (320:dt:T_end);
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
                X=[];
                for trial_index=1:size(trainingData,1)
                    X=[X mean(trainingData(trial_index,angle_index).spikes(:,1:T(time_index)),2)];
                end
                Xt=[];
                for trial_index=1:size(testData,1)
                    Xt=[Xt mean(testData(trial_index,angle_index).spikes(:,1:T(time_index)),2)];
                end
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
r_array = r_vec;
RMSE_array_mean = mean(rmse,2)';
RMSE_array_std = std(rmse,[],2)'
% r_array = [50, r_array];

figure;
scatter(r_array, RMSE_array_mean);
hold on
errorbar(r_array, RMSE_array_mean, RMSE_array_std, 'LineStyle', 'none', 'Color', 'k');

hold on
y = zeros(1, length(r_array));
std_y = zeros(1, length(r_array));
y_plus = zeros(1, length(r_array));
y_minus = zeros(1, length(r_array));

for i = 1: length(r_array)
    y(1,i) = 6.82;
    std_y(1,i) = 0.49;
    y_plus(1,i) = y(1,i) + std_y(1,i);
    y_minus(1,i) = y(1,i) - std_y(1,i);
end
plot(y, 'g', 'LineWidth', 1.5);
hold on

plot(r_array, y_plus, 'r--', 'LineWidth', 1.5);
hold on
plot(r_array, y_minus, 'r--', 'LineWidth', 1.5);
xlim([1 70]);

xlabel("r")
ylabel("rmse")