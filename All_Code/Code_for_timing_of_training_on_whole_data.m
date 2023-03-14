load 'monkeydata_training.mat'
t=[];
for k=1:50
    tic
    positionEstimatorTraining(trial)
    t(k)=toc
end
t_mean=mean(t);
t_std=std(t);