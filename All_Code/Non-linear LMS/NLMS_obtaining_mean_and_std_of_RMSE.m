%%  Using Nonlinear LMS
clc
close all
clear all

%% Loading Data
load monkeydata_training.mat;
N_iter=50;
RMSE=zeros(N_iter,1);
t_tot=0;

for iter=1:N_iter
    disp("Iteration #"+ iter)
    tic
%% Obtaining Training and Test sets

%Random permutations of 100 trials
ix = randperm(length(trial));
split = 70; %As percentage

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:split),:);
testData = trial(ix(split+1:end),:);

% - trainingData:
%     trainingData(n,k)              (n = trial id,  k = reaching angle)
%     trainingData(n,k).trialId      unique number of the trial
%     trainingData(n,k).spikes(i,t)  (i = neuron id, t = time)
%     trainingData(n,k).handPos(d,t) (d = dimension [1-3], t = time)

%% Extracting feature vectors at each time step for each trial
T_end = 560;
dt=20;
[feat,l,t]=organize_data(trainingData,dt,T_end);
T = (320:dt:T_end);
feature_struct_trials = struct;

for angle_ind = 1:8
    for T_ind = 1:length(T)
        temp = feat(t<=T(T_ind),l==angle_ind);
        feature_struct_trials(T_ind,angle_ind).feature_vector=temp;        
    end
end

%% Sample hand positions for each angle
%Extract all hand positions
[mx,my,x,y,len,in_data]=get_all_handPos(trainingData);

%Re-sample hand position data at T = (320:dt:560);
mx_resampled = mx(:,T,:);
my_resampled = my(:,T,:);
x_resampled = x(:,T,:);
y_resampled = y(:,T,:);

%% Considering all angles - Implementing LMS
angle =1:8;
N=size(x,1); %Number of time-step-trials for LMS.
mu = 0.02; %Learning rate for LMS
rho=0.007;
est_hand_pos = zeros(length(T),length(angle));
N_epochs = 500;
Pre_training = round(N/3); %Number of samples.

% figure;
for angle_index =1:length(angle)
    features_for_LMS = feature_struct_trials(:,angle_index);
    handPos_x_for_LMS = x_resampled(:,:,angle_index);
    handPos_y_for_LMS = y_resampled(:,:,angle_index);
    
    for time_index = 1:length(T)
        features_temp = features_for_LMS(time_index).feature_vector;
        features_temp = [ones(1,size(features_temp,2)); features_temp];
        hand_pos_temp_x = handPos_x_for_LMS(:,time_index);
        [hand_pos_temp_x, a_x, b_x] = normalise_matrix(hand_pos_temp_x,-1,1);
        hand_pos_temp_y = handPos_y_for_LMS(:,time_index);
        [hand_pos_temp_y, a_y, b_y] = normalise_matrix(hand_pos_temp_y,-1,1);
        
        L = length(features_temp);
        w_init_x = zeros(L,1);
        w_init_y = zeros(L,1);
        
        %Pre-training weights
        for epoch_ind = 1:N_epochs
            % Implementing LMS for prediction for first 10 samples
            [~,~,wpre_x,~,~] = nonlinear_LMS_final(features_temp,hand_pos_temp_x,mu,Pre_training,w_init_x);
            [~,~,wpre_y,~,~] = nonlinear_LMS_final(features_temp,hand_pos_temp_y,mu,Pre_training,w_init_y);
            w_init_x = wpre_x(:,end);
            w_init_y = wpre_y(:,end);
        end
        
        %Implementing NLMS algorithm
        [estimated_hand_pos_x,~,w_x,lambda_x,s_x] = nonlinear_LMS_final(features_temp,hand_pos_temp_x,mu,N,w_init_x);
        [estimated_hand_pos_y,~,w_y,lambda_y,s_y] = nonlinear_LMS_final(features_temp,hand_pos_temp_y,mu,N,w_init_y);
        est_hand_pos_x(time_index,angle_index) = estimated_hand_pos_x(end);
        est_hand_pos_x(time_index,angle_index) = (est_hand_pos_x(time_index,angle_index) - b_x)./(a_x); 
        est_hand_pos_y(time_index,angle_index) = estimated_hand_pos_y(end);
        est_hand_pos_y(time_index,angle_index) = (est_hand_pos_y(time_index,angle_index) - b_y)./(a_y); 
        
        %Storing coefficients
        est_w_x(time_index,angle_index).coeff_x = w_x(:,end);
        est_lambda_x(time_index,angle_index).coeff_x = lambda_x(end);
        est_a_x(time_index,angle_index) = a_x;
        est_b_x(time_index,angle_index) = b_x;
        est_w_y(time_index,angle_index).coeff_y = w_y(:,end);
        est_lambda_y(time_index,angle_index).coeff_y = lambda_y(end);
        est_a_y(time_index,angle_index) = a_y;
        est_b_y(time_index,angle_index) = b_y;
    end
%     plot(handPos_x_for_LMS',handPos_y_for_LMS','b')
%     hold on
%     plot(est_hand_pos_x(:,angle_index),est_hand_pos_y(:,angle_index),'r')
end

%% Apply model obtained on test data

%Extract feature vectors
[feat,l,t]=organize_data(testData,dt,T_end);
feature_struct_test = struct;
for angle_ind = 1:8
    for T_ind = 1:length(T)
        temp = feat(t<=T(T_ind),l==angle_ind);
        feature_struct_test(T_ind,angle_ind).feature_vector=temp;        
    end
end

%% Extract all hand positions - Useful for checking error
[mx_test,my_test,x_test,y_test,len_test,in_data_test]=get_all_handPos(testData);
%Re-sample hand position data at T = (320:dt:560);
x_test_resampled = x_test(:,T,:); 
mx_test_resampled = mx_test(:,T,:); 
y_test_resampled = y_test(:,T,:);
my_test_resampled = my_test(:,T,:);
% % Make hand position into complex data
% hand_pos_complex_test = x_test_resampled + 1j*y_test_resampled; 

%% Using coefficients to obtain hand position
N_test_trials = size(x_test,1);
est_hand_pos_test_x = zeros(N_test_trials,length(T),length(angle));
est_hand_pos_test_y = zeros(N_test_trials,length(T),length(angle));
% figure;
for n=1:N_test_trials
    for angle_index =1:length(angle)
        for T_ind = 1:length(T)
            features_temp = feature_struct_test(T_ind,angle_index).feature_vector(:,n);
            features_temp = [ones(1,size(features_temp,2)); features_temp];
            est_hand_pos_test_x(n,T_ind,angle_index) = (est_lambda_x(T_ind,angle_index).coeff_x)*tanh((est_w_x(T_ind,angle_index).coeff_x)'*features_temp);
            est_hand_pos_test_x(n,T_ind,angle_index) = (est_hand_pos_test_x(n,T_ind,angle_index) - est_b_x(T_ind,angle_index))./ est_a_x(T_ind,angle_index);
            est_hand_pos_test_y(n,T_ind,angle_index) = (est_lambda_y(T_ind,angle_index).coeff_y)*tanh((est_w_y(T_ind,angle_index).coeff_y)'*features_temp); 
            est_hand_pos_test_y(n,T_ind,angle_index) = (est_hand_pos_test_y(n,T_ind,angle_index) - est_b_y(T_ind,angle_index))./ est_a_y(T_ind,angle_index);
        end
        
%         plot(x_test_resampled(n,:,angle_index),y_test_resampled(n,:,angle_index),'b');
%         hold on
%         plot(est_hand_pos_test_x(n,:,angle_index),est_hand_pos_test_y(n,:,angle_index),'r');
    end
end

% for k=1:8
%     plot(mx_test_resampled(:,:,k),my_test_resampled(:,:,k),'g','Linewidth',1)
% end

%% Finding RMSE given the angle
xt_r=x_test(:,320:20:end,:);
yt_r=y_test(:,320:20:end,:);
in_test_r=in_data_test(:,320:20:end,:);
mx_r=mx(:,320:20:end,:);
my_r=my(:,320:20:end,:);
[predMSE,meanMSE]=mse_v_time_final(est_hand_pos_test_x,est_hand_pos_test_y,xt_r,yt_r,mx_r,my_r,in_test_r);

predMSE_vec(iter,:) = predMSE(1:13);
meanMSE_vec(iter,:) = meanMSE(1:13);
RMSE_pred(iter) = sqrt(sum(predMSE))
RMSE_mean_traj(iter) = sqrt(sum(meanMSE))


t_elapsed(iter)=toc
t_tot=t_tot+t_elapsed(iter);
p_complete=iter/N_iter;
disp(round(100*p_complete,2)+"% complete")
disp(round((t_tot/p_complete-t_tot)/60,1)+" minutes left (estimated)")
end

figure;
subplot(1,2,1)
plot(mean(predMSE_vec,1))
hold on
plot(mean(meanMSE_vec,1))
legend('Using non-linear LMS','Using mean trajectory')

subplot(1,2,2) %No plot when iter=1, as it takes variance across columns which is a scalar
plot(var(predMSE_vec,1))
hold on
plot(var(meanMSE_vec,1))
legend('Using non-linear LMS','Using mean trajectory')

disp("Mean RMSE value using NLMS = "+mean(RMSE_pred))
disp("Std of RMSE value using NLMS = "+std(RMSE_pred))
disp("Mean RMSE value using Mean Tajectory = "+mean(RMSE_mean_traj))
disp("Std of RMSE value using Mean Tajectory = "+std(RMSE_mean_traj))

