
clc;clear;close all;
load 'monkeydata_training.mat';

%Split dataset into train and test
trial_num = size(trial,1);
train_pertentage = 0.8;
train_num = round(train_pertentage*size(trial,1));
temp = randperm(trial_num);

train_index = temp(1:train_num);
train_data=trial(train_index,:);

if train_num < trial_num
    test_index=temp((train_num+1):end);
    test_data = trial(test_index,:);
else
    test_data=[];
end



