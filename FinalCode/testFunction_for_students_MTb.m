%% Continuous Position Estimator Test Script with timing
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

clear all; clc;
% Start overall timing
totalTimeStart = tic;

% Load training data
fprintf('Loading data...\n');
load("monkeydata_training.mat");

% Set random number generator for reproducibility
rng(2013);
ix = randperm(length(trial));

% Select training and testing data (50 trials for training, rest for testing)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...\n');

% Initialize error tracking
meanSqError = 0;
n_predictions = 0;  

% Create figure for visualization
figure;
hold on;
axis square;
grid on;

% Start training timing
trainingTimeStart = tic;
fprintf('Training model...\n');

% Train Model using positionEstimatorTrainingTest
modelParameters = positionEstimatorTrainingTest(trainingData);

% End training timing
trainingTime = toc(trainingTimeStart);
fprintf('Training completed in %.2f seconds\n', trainingTime);

% Start testing timing
testingTimeStart = tic;
fprintf('Starting decoding...\n');

% Keep track of per-trial timing
trialTimes = zeros(size(testData, 1), 1);

% Loop through test trials
for tr = 1:size(testData, 1)
    trialTimeStart = tic;
    fprintf('Decoding block %d out of %d\n', tr, size(testData, 1));
    
    % Process each direction (randomized order)
    for direc = randperm(8) 
        decodedHandPos = [];
        
        % Define time points for testing (20ms steps starting at 320ms)
        times = 320:20:size(testData(tr,direc).spikes,2);
        
        % Loop through each time point
        for t = times
            % Create data structure for this time point
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;
            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            % Get position estimate using our decoder
            [decodedPosX, decodedPosY] = positionEstimatorTest(past_current_trial, modelParameters);
            
            % Store result
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            % Calculate squared error
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
        end
        
        % Count predictions and plot results
        n_predictions = n_predictions + length(times);
        plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times), testData(tr,direc).handPos(2,times), 'b');
    end
    
    % Record trial time
    trialTimes(tr) = toc(trialTimeStart);
    fprintf('Trial %d completed in %.2f seconds\n', tr, trialTimes(tr));
end

% End testing timing
testingTime = toc(testingTimeStart);
fprintf('Decoding completed in %.2f seconds (%.2f sec/trial avg)\n', testingTime, mean(trialTimes));

% Add legend and calculate RMSE
legend('Decoded Position', 'Actual Position');

% Calculate overall RMSE
RMSE = sqrt(meanSqError/n_predictions);
fprintf('RMSE = %.4f cm\n', RMSE);

% End overall timing and print summary
totalTime = toc(totalTimeStart);
fprintf('\n==== Performance Summary ====\n');
fprintf('Total execution time: %.2f seconds\n', totalTime);
fprintf('Training time: %.2f seconds (%.1f%%)\n', trainingTime, (trainingTime/totalTime)*100);
fprintf('Testing time: %.2f seconds (%.1f%%)\n', testingTime, (testingTime/totalTime)*100);
fprintf('Average time per trial: %.2f seconds\n', mean(trialTimes));
fprintf('Average time per prediction: %.4f seconds\n', testingTime/n_predictions);