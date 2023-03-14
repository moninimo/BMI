clc;clear;close all;
load monkeydata_training.mat;
max_iter=100;
for iter=1:max_iter
    iter
    tic
    ix = randperm(length(trial));
    trainingData = trial(ix(1:70),:);
    testData = trial(ix(71:end),:);
    [avgx,avgy,x,y,~,~]=get_all_handPos(trainingData);
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
                temp1=(xt(i,j,k)-avgx(1,min(j,size(avgx,2)),k))^2+(yt(i,j,k)-avgy(1,min(j,size(avgy,2)),k))^2;
                er1=er1+temp1*id(i,j,k);
            end
        end
    end
    rmse_avg(iter)=sqrt(er1/sum(sum(sum(id))));
    toc
end
[mean(rmse_avg) std(rmse_avg)]