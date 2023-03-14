%%Script to run the final algorithm with different sp
%%when "for_averaging"=1, then 50 iteration are set and
%%no plots are printed. Otherwise, one iteration is performed
%%with the trajectory outputs plotted
clc;clear;close all;
load monkeydata_training.mat;
sp=0.7;
for_averaging=0;

max_iter=2;
if for_averaging~=1
    max_iter=1;
end

for iter=1:max_iter
    iter
    tic
    ix = randperm(length(trial));
    trainingData = trial(ix(1:(100*sp)),:);
    testData = trial(ix((100*sp+1):end),:);
    RMSE(iter) = testFunction_for_students_MTb(trainingData,testData,for_averaging);
    toc
end

[mean(RMSE) std(RMSE)]
%%
function RMSE = testFunction_for_students_MTb(trainingData,testData,for_averaging)
meanSqError = 0;
n_predictions = 0;
if for_averaging~=1
    figure
    hold on
    axis square
    grid
end

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

for tr=1:size(testData,1)
    if for_averaging~=1
        display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
        pause(0.001)
    end
    for direc=randperm(8)
        decodedHandPos = [];
        
        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t);
            past_current_trial.decodedHandPos = decodedHandPos;
            
            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        if for_averaging~=1
            hold on
            plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
            plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
        end
    end
end
if for_averaging~=1
    legend('Decoded Position', 'Actual Position')
end
RMSE = sqrt(meanSqError/n_predictions);
end
