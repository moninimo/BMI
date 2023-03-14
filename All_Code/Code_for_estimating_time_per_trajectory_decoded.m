%%Script for estimating time per trajectory predicted

%Use the whole dataset which contains 800 trajectories for both training and testing
%Obviously, since we train and test on the same dataset the RMSE result is
%not representative of the efffectiveness of the algorith
%However, the time should not be affected. So we can run this and then
%divide by 800 to get an estimate of the time taken to predict each
%trajectory
clc;clear;close all
load 'monkeydata_training.mat'
tic
[modelParameters] = positionEstimatorTraining(trial);
toc
max_iter=20;
for iter=1:max_iter
    tic
    iter
    for tr=1:size(trial,1)
        for direc=randperm(8)
            decodedHandPos = [];
            times=320:20:size(trial(tr,direc).spikes,2);
            for t=times
                past_current_trial.trialId = trial(tr,direc).trialId;
                past_current_trial.spikes = trial(tr,direc).spikes(:,1:t);
                past_current_trial.decodedHandPos = decodedHandPos;
                past_current_trial.startHandPos = trial(tr,direc).handPos(1:2,1);
                if nargout('positionEstimator') == 3
                    [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                    modelParameters = newParameters;
                elseif nargout('positionEstimator') == 2
                    [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
                end
                decodedPos = [decodedPosX; decodedPosY];
                decodedHandPos = [decodedHandPos decodedPos];
            end
        end
    end
    time_for_800(iter)=toc
end
mean(time_for_800/800)