
clc;clear;close all;
load 'monkeydata_training.mat';

%Split dataset into train and test
trial_num = size(trial,1);
train_pertentage = 0.7;
train_num = round(train_pertentage*size(trial,1));
temp = randperm(trial_num);
NN_vec=1:10:261;

train_index = temp(1:train_num);
train_data=trial(train_index,:);

if train_num < trial_num
    test_index=temp((train_num+1):end);
    test_data = trial(test_index,:);
else
    test_data=[];
end

[X,l,~,mX]=Exfeature(train_data);
[Xt,l_test,~,~]=Exfeature(test_data);

pred=kNN_kernal(2,Xt,mX,1:8,1);
pred=[pred; kNN_kernal(2,Xt,X,l,NN_vec)];

acc0=100*(mean(pred==l_test,2));

figure;
plot(acc0)


