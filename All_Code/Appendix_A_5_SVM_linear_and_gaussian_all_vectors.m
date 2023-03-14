close all
clear all
clc
load('monkeydata_training.mat');

s2_vec=[0.1 0.5 1 5 10];
c_vec=[0.1 1 10 100 500];
for i=1:10
    i
    tic
    dt=20;
    %Random permutations of 100 trials
    ix = randperm(length(trial));
    split = 70; %As percentage
    
    % Select training and testing data (you can choose to split your data in a different way if you wish)
    train = trial(ix(1:split),:);
    test= trial(ix(split+1:end),:);
    
    [F,l,t,mF]=organize_data1(train,dt,320);
    [F_test,l_test,t_test]=organize_data1(test,dt,320);

    for c_ind=1:length(c_vec)
        for s_ind=1:length(s2_vec)
            pred = do_SVM_gaussian_all_features(F_test,F,l,c_vec(c_ind),s2_vec(s_ind));
            acc(s_ind,c_ind,i) = mean(pred==l_test)*100;
        end
        pred = do_SVM_linear_all_features(F_test,F,l,c_vec(c_ind));
        acc(s_ind+1,c_ind,i) = mean(pred==l_test)*100;
    end
    toc
end
%%
acc_mean=mean(acc,3);
acc_std=std(acc,[],3);
figure(1)
h=heatmap(acc_mean);
h.Colormap=bone;
h.XData=c_vec;
h.YData=[num2str([s2_vec]');"LINEAR"];
xlabel("C")
ylabel("Sigma")

figure(2)
h=heatmap(acc_std);
h.XData=c_vec;
h.Colormap=bone;
h.YData=[num2str([s2_vec]');"LINEAR"];
xlabel("C")
ylabel("Sigma")