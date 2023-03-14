close all
clear all
clc

load('monkeydata_training.mat');
c_vec = 1;

for i=1:20
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
    pred = do_SVM_linear(F_test,mF,c_vec(c_ind));
    acc(i,c_ind) = mean(pred==l_test)*100;
end
%%

toc
end

%%
mean(acc)
std(acc)