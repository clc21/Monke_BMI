% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

% function RMSE = testFunction_for_students_MTb(teamName)

% load monkeydata0.mat

clear all; clc;
% Start overall timing
totalTimeStart = tic;

% Load training data
fprintf('Loading data...\n');
load("monkeydata_training.mat");

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  


figure
hold on
axis square
grid

% Start training timing: Not in actual file
trainingTimeStart = tic;
fprintf('Training model...\n');

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

% End training timing: Not in actual file
trainingTime = toc(trainingTimeStart);
fprintf('Training completed in %.2f seconds\n', trainingTime);

% Start testing timing: Not in actual file
testingTimeStart = tic;
fprintf('Starting decoding...\n');

% Keep track of per-trial timing:
trialTimes = zeros(size(testData, 1), 1);

for tr=1:size(testData,1)
    trialTimeStart = tic;
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
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
        hold on
        % Commented out trajectory plot
        % plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        % plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
    % Record trial time
    trialTimes(tr) = toc(trialTimeStart);
    fprintf('Trial %d completed in %.2f seconds\n', tr, trialTimes(tr));
end

% End testing timing
testingTime = toc(testingTimeStart);
fprintf('Decoding completed in %.2f seconds (%.2f sec/trial avg)\n', testingTime, mean(trialTimes));

% legend('Decoded Position', 'Actual Position')


RMSE = sqrt(meanSqError/n_predictions);
fprintf('RMSE = %.4f cm\n', RMSE);

% rmpath(genpath(teamName))
% End overall timing and print summary
totalTime = toc(totalTimeStart);
fprintf('\n==== Performance Summary ====\n');
fprintf('Total execution time: %.2f seconds\n', totalTime);
fprintf('Training time: %.2f seconds (%.1f%%)\n', trainingTime, (trainingTime/totalTime)*100);
fprintf('Testing time: %.2f seconds (%.1f%%)\n', testingTime, (testingTime/totalTime)*100);
fprintf('Average time per trial: %.2f seconds\n', mean(trialTimes));
fprintf('Average time per prediction: %.4f seconds\n', testingTime/n_predictions);
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters)
    % positionEstimatorTest - Decode hand position from neural data for testing
    %
    % This function predicts the hand position given the neural data and trained models.
    % It first classifies the reaching angle using KNN, then uses the corresponding
    % Kalman filter to predict the hand trajectory.
    %
    % Input:
    %   past_current_trial - Structure containing neural data and past decoded positions
    %   modelParameters - Structure containing trained models
    %
    % Output:
    %   decodedPosX - Decoded X position
    %   decodedPosY - Decoded Y position
    
    % Persistent variables to maintain state between calls
    persistent predictedAngle
    
    % Set same parameter for spike rate binning as training
    binSize = 20;                 % 20ms bins
    windowSize = 160;             % window in ms
    nBins = windowSize / binSize;  % 8 bins

    % Find the last 160 ms test data
    endIdx = size(past_current_trial.spikes, 2);  % current time
    startIdx = max(1, endIdx - windowSize + 1);   % go back 160 ms 
    
    % Truncate test data for dimension compatibility
    spikeWindow = past_current_trial.spikes(:, startIdx:endIdx);
    spikeWindow_reshaped = reshape(spikeWindow, size(spikeWindow, 1), binSize, nBins);

    % Sum over time axis to get 98 × 8 binned spike rate matrix
    binnedSpikes = sum(spikeWindow_reshaped, 2)/ binSize; % sum along 2nd dim
    % Squeeze to remove singleton dimension → 98 × 8
    binnedSpikes = squeeze(binnedSpikes);
    
    % smooth and center spike rate
    spikeRate_test = applyGaussianFilter(binnedSpikes);
    x_flat = spikeRate_test(:); % → 784 × 1
    x_centered = x_flat - modelParameters.meanVec;  

    % project into MDF space
    W_test = modelParameters.projMat' * x_centered;
  
    predictedAngle = predict(modelParameters.bestKNNModels, W_test');
    % Ensure valid angle
    predictedAngle = max(1, min(8, predictedAngle));

    % Initialize Kalman filter with the right model
    Kalman = KalmanFilterRegression(n_neurons=98, alpha=0.1, binSize=10, delaySteps=0);
    % Kalman.setInitialPos(past_current_trial.startHandPos);
    Kalman.addModel(modelParameters.kalmanModels{predictedAngle});
    
    % Extract features from current spike data
    [spikes, ~, ~] = extractFeatures(past_current_trial, isStruct=false, winSz=10, winStp=10);
    % totalTime = size(spikes, 2);
    % spikeRate = spikes(:,startInd:totalTime);
    % startInd = totalTime+1;
    Kalman.setInitialPos(past_current_trial.startHandPos);
    % Update Kalman filter prediction
    Kalman.predict(spikes);
    [decodedPosX, decodedPosY] = Kalman.getHandPos();
    
    % Update decoded position for next call
    decodedPos = [decodedPosX; decodedPosY];
end

function [modelParameters] = positionEstimatorTraining(trainingData)
    % positionEstimatorTraining - Train models for BMI neural decoding
    % UPDATED TO: KNN IMPROVEMENT
    % This function trains a KNN model for angle classification and a set of
    % Kalman filters for trajectory prediction, one for each reaching angle.
    %
    % Input:
    %   trainingData - Trials data for training
    %
    % Output:
    %   modelParameters - Structure containing trained models
    
    % Create KNN model for angle classification
    fprintf('Training KNN model for angle classification...\n');
    
    % Filter out the 300ms before and 100ms after trial data 
    filtered_data = filterTrial(trainingData, 'min');
    
    % Extract spike rate and apply Gaussian Smoothing to it
    [~, ~, spikeRateMat] = getSmoothedSpikeRate(filtered_data);
    % Reformat the spike rate matrix to suit PCA 
    %(numNeurons*timebins x numTrials*angle)
    X_pca = inputPCA(spikeRateMat);
    
    % Apply PCA to the reformated spike rate matrix
    [eigVectors, afterPCA, meanVec] = applyPCA(X_pca);
    % Perform LDA onto the principle components to get the mean position of
    % the neurons
    [W, projMat] = applyLDA(afterPCA, X_pca, filtered_data);
    
    numClasses = 8;
    samplesPerClass = 50;  % Adjusted to match 400 samples total
  
    % Section below is creating singular best knn model
    % Define custom labels
    customLabels = [1, 2, 3, 4, 5, 6, 7, 8];
    
    % Initialize the label matrix
    labelMatrix = zeros(numClasses * samplesPerClass, 1);
    
    % Assign custom labels to each class
    for i = 1:numClasses
        labelMatrix((i-1)*samplesPerClass + 1:i*samplesPerClass) = customLabels(i);
    end
    
    X_train = W';               % 400 × d (d = MDF dimension)
    y_train = labelMatrix;      % 400 × 1
    
    % Create a knn model
    best_k = 5; % best_k = 5 based on Overleaf report
    bestKNNModels = fitcknn(X_train, y_train, 'NumNeighbors', best_k);
    
    % Initialize array to store Kalman models for each angle
    fprintf('Training Kalman filters for trajectory prediction...\n');
    modelParametersKalman = cell(1, 8);
    
    % Train a separate Kalman filter for each reaching angle
    for angle = 1:8
        fprintf('Training Kalman filter for angle %d/8...\n', angle);
        
        % Initialize Kalman filter for this angle
        Kalman = KalmanFilterRegression(n_neurons=98, alpha=0.1, binSize=10, delaySteps=0);
        
        % Train on all trials for this angle
        for t = 1:size(trainingData, 1)
            % Extract features from this trial
            [spikeRate, handKinematics, ~] = extractFeatures(trainingData, trialNumber=t, angle=angle, isStruct=true, winSz=10, winStp=10);
            
            % Set initial position and train
            Kalman.setInitialPos(handKinematics(1:2, 1));
            Kalman.fit(spikeRate, handKinematics);
            Kalman.predict(spikeRate, handKinematics);
        end
        
        % Clear RMSE data and store the trained model
        Kalman.clearRMSe();
        
        % Store the trained model for this angle
        modelParametersKalman{angle} = Kalman.model;
    end

% Return model parameters
modelParameters = struct();
modelParameters.kalmanModels = modelParametersKalman; % Kalman models for each angle
modelParameters.W = W;                        % LDA projection matrices
modelParameters.projMat = projMat;            % Projected features after LDA
modelParameters.meanVec = meanVec;            % PCA mean vectors
modelParameters.bestKNNModels = bestKNNModels;        % Trained KNN models for each time step
fprintf('Model training completed successfully.\n');
end

%%%%%%%%%%%%%%%%%%%% Function %%%%%%%%%%%%%%%%%%%%%

function [filtered_data] = filterTrial(data, matchLengthMethod)
    arguments
        data struct
        matchLengthMethod char {mustBeMember(matchLengthMethod, {'max', 'min'})} = 'max'
    end
    
    % Check if the data has a 'trial' field
 % Check if data is already a structure with trial field
    if isfield(data, 'trial')
        data = data;
    else
        % Create a temporary structure if data doesn't have trial field
        data = struct('trial', data);
    end
    
    % Get dimensions of the trial struct array
    [n_trials, n_angles] = size(data.trial);
    
    % Initialize output with the same structure as input
    filtered_data = data;
    
    % Preallocate trial_lengths for better performance
    trial_lengths = zeros(n_trials * n_angles, 1);
    lengths_count = 0;
    
    % Determine all valid lengths
    for trial_idx = 1:n_trials
        for angle_idx = 1:n_angles
            total_time = size(data.trial(trial_idx, angle_idx).spikes, 2);
            start_idx = 301; 
            end_idx = total_time - 100;
            
            if end_idx > start_idx
                lengths_count = lengths_count + 1;
                trial_lengths(lengths_count) = end_idx - start_idx + 1;
            end
        end
    end
    
    % Trim any unused elements in the array
    trial_lengths = trial_lengths(1:lengths_count);
    
    % Get target length in one operation
    if strcmp(matchLengthMethod, 'max')
        target_length = max(trial_lengths);
    else
        target_length = min(trial_lengths);
    end
    
    % Process all trials
    for trial_idx = 1:n_trials
        for angle_idx = 1:n_angles
            current_trial = data.trial(trial_idx, angle_idx);
            total_time = size(current_trial.spikes, 2);
            start_idx = 301;
            end_idx = total_time - 100;
            
            if end_idx <= start_idx
                continue;
            end
            
            % Process spike data
            if isfield(current_trial, 'spikes')
                extracted_spikes = current_trial.spikes(:, start_idx:end_idx);
                [num_channels, current_length] = size(extracted_spikes);
                
                if current_length < target_length
                    % Pad with zeros - preallocate for speed
                    adjusted_spikes = zeros(num_channels, target_length);
                    adjusted_spikes(:, 1:current_length) = extracted_spikes;
                else
                    % Truncate to target length
                    adjusted_spikes = extracted_spikes(:, 1:target_length);
                end
                
                filtered_data(trial_idx, angle_idx).spikes = adjusted_spikes;
            end
            
            % Process handPos data
            if isfield(current_trial, 'handPos')
                extracted_handPos = current_trial.handPos(:, start_idx:end_idx);
                [num_channels, current_length] = size(extracted_handPos);
                
                if current_length < target_length
                    % Pad with zeros - preallocate for speed
                    adjusted_handPos = zeros(num_channels, target_length);
                    adjusted_handPos(:, 1:current_length) = extracted_handPos;
                else
                    % Truncate to target length
                    adjusted_handPos = extracted_handPos(:, 1:target_length);
                end
                
                filtered_data(trial_idx, angle_idx).handPos = adjusted_handPos;
            end
        end
    end
end

function [X_pca] = inputPCA(spikeRateData)

    arguments
        spikeRateData cell
    end

    [n, k] = size(spikeRateData);                       % n = num_trials, k = angles
    [numFeatures, timeBins] = size(spikeRateData{1,1}); % Get number of features (number of neurons)
    X_pca = zeros(numFeatures*timeBins, n*k);           % Preallocate matrix (800x784)

    
    for angle = 1:k
        for t = 1:n
            % Extract spike rate matri (nunNeurons x timebins)
            spikeRate = spikeRateData{t, angle};
            % Flatten the matric into a column vector
            flattenedVector = spikeRate(:);
            % Compute the column index in X_pca
            colIdx = (angle-1)*n + t;
            % Assign flattened vector to the corresponding column
            X_pca(:, colIdx) = flattenedVector;
            
        end
    end
    
end
function [pcaParams_mat, reducedFeatures_mat, spikeRateMat] = getSmoothedSpikeRate(data)
    arguments
        data struct
    end
    
    [n,k] = size(data);                % n = num_trials, k = angles
    pcaParams_mat = cell(n, k);        % Store PCA parameters for each angle
    reducedFeatures_mat = cell(n, k);  % Store PCA-reduced features
    spikeRateMat = cell(n, k);

    for angle = 1:k
        for t = 1:n
            % Call extractWindows with all required arguments and grab
            % spike rate matrix
            [spikeRate, ~] = extractWindows(data, t, angle);

            % Apply gaussian smoothing
            spikeRate = applyGaussianFilter(spikeRate);

            % Store spikeRate into a struct
            spikeRateMat{t, angle} = spikeRate;

            % Apply PCA
            %[pcaParams, reducedFeatures] = obj.applyPCA(spikeRate);

            % Store the results
            %pcaParams_mat{t, angle} = pcaParams;
            %reducedFeatures_mat{t, angle} = reducedFeatures;
 
        end
    end
end

function [spikeRate, time_bins] = extractWindows(trial, trialNumber, angle, args)
%extractWindows Extract the spike rate in time windows
%   trial
arguments
    trial struct
    trialNumber double
    angle double
    args.isStruct logical = true
    args.winSz double = 20  % ms
    args.winStp double = 20

end

if args.isStruct
    spike_train = trial(trialNumber, angle).spikes;
else
    spike_train = trial.spikes;
end

n_neurons = size(spike_train, 1);  % Number of neurons
n_time = size(spike_train, 2);     % Total time points
n_wind = floor((n_time - args.winSz) / args.winStp) + 1; % Number of windows

% Initialize arrays
spikeRate = zeros(n_neurons, n_wind);
time_bins = zeros(1, n_wind);

% Sliding window loop
for w = 1:n_wind
    % Define window range
    t_start = (w - 1) * args.winStp + 1;
    t_end = t_start + args.winSz - 1;

    % Compute spike rate (in Hz)
    spikeRate(:, w) = sum(spike_train(:, t_start:t_end), 2) / args.winSz; 

    % Store time bin center
    time_bins(w) = round((t_start + t_end) / 2);
end
end

function [eigVectors,reducedFeatures, meanVec] = applyPCA(X)

% IMPLEMENTATION : [pcaParams, reducedFeatures] = applyPCA(spikeRates);
% INPUT: m by n matrix - observations (n samples) and number of features (m variables)
%        
% OUTPUT: reducedFeatures --> principal components

    % Normalize data by subtract cross-trial mean
    X = X - mean(X, 2);
    meanVec = mean(X,2);
    % Compute covariance matrix
    covMatrix = (X' * X)/(size(X, 2));

    % Compute eigenvalues and eigenvectors
    [eigVectors, eigValues] = eig(covMatrix);

    % Sort eigenvalues in descending order
    [~, sortIdx] = sort(diag(eigValues), 'descend');
    eigVectors = eigVectors(:, sortIdx);
    
    % Get eigenvalues and not the zeros of the diagonal matrix
    eigValues = diag(eigValues);
    eigValues = diag(eigValues(sortIdx));

    % project data onto the newly derived basis and normalise it
    pc = X*eigVectors;
    reducedFeatures = pc./sqrt(sum(pc.^2));

end
    
function [W, projMat] = applyLDA(reducedFeatures, X_pca, data)

    [n, k] = size(data);
    dataMean = zeros(size(X_pca,1),k);
    % Get between-class and within-class scatter matrices

    for angle = 1: k
        dataMean(:,angle) =  mean(X_pca(:,n*(angle-1)+1:angle*n),2);
    end

    Sb = (dataMean - mean(X_pca,2))*(dataMean - mean(X_pca,2))'; % Between class scatter matrix
    x_grand =  (X_pca - mean(X_pca,2))*(X_pca - mean(X_pca,2))'; % Grand mean
    Sw = x_grand - Sb; % Within class scatter matrix
    
    pcaDim = 300; % Keep the first 200 components from PCA
    Sw = reducedFeatures(:,1:pcaDim)' * Sw * reducedFeatures(:,1:pcaDim);
    Sb = reducedFeatures(:,1:pcaDim)' * Sb * reducedFeatures(:,1:pcaDim);

    % Ensure Sw is invertible
    if rank(Sw) < size(Sw, 1)
        Sw = pinv(Sw);  % Use pseudo-inverse if singular
    else
        Sw = inv(Sw);   % Regular inverse
    end

    [eigVecLDA, eigValLDA] = eig(Sw*Sb);

    [~,sortIdx] = sort(diag(eigValLDA),'descend');
    ldaDim = 7; % arbitraty atm
    % optimum output
    projMat = reducedFeatures(:,1:pcaDim)*eigVecLDA(:,sortIdx(1:ldaDim));
    % optimum projection from the Most Discriminant Feature Method....!
    W = projMat' * (X_pca - mean(X_pca,2));
end

function smoothedSpikeRate = applyGaussianFilter(spikeRate, sigma, winSz)
arguments
    spikeRate
    sigma double = 50  % Standard deviation for Gaussian smoothing (in ms)
    winSz double = 10  % Window size in ms
end
    % Create Gaussian kernel
    kernelSize = ceil(3 * sigma / winSz) * 2 + 1;  % Ensure it's odd
    t = linspace(-kernelSize / 2, kernelSize / 2, kernelSize);
    gaussKernel = exp(-t.^2 / (2 * sigma^2));
    gaussKernel = gaussKernel / sum(gaussKernel);  % Normalize

    % Apply convolution along the time dimension (2nd axis)
    smoothedSpikeRate = conv2(spikeRate, gaussKernel, 'same');
end


function [Ypred, accuracy, best_k] = Pred_accuracy(X, y, Xnew, Ynew) % calculate the best k and predict and calculate accuracy angle classification
    arguments
        X    % Feature vector of training data
        y    % Labels (angle)
        Xnew % Feature vector of test data
        Ynew % Labels of test data
    end

    % Select the optimal k (best_k)
    k_val = 1:10;
    cv_errors = zeros(size(k_val));

    for i = 1:length(k_val)
        knnModel = fitcknn(X, y, 'NumNeighbors', k_val(i), 'CrossVal', 'on', 'KFold', 10);
        % y_pred = predict(model, X_test);
        cv_errors(i) = kfoldLoss(knnModel); % compute cross-validation error
    end

    % Find the best k (minimum error)
    [~, best_k_idx] = min(cv_errors);
    best_k = k_val(best_k_idx);
    % Ensure k is an odd number
    if mod(best_k, 2) == 0
        best_k = best_k + 1;  
    end

    % Train final knn model with best_k
    best_knnModel = fitcknn(X, y, 'NumNeighbors', best_k);
    Ypred = predict(best_knnModel, Xnew);

    % Accuracy
    y = y(:);      % Convert y to column vector
    Ynew = Ynew(:); % Convert Ynew to column vector
    accuracy = sum(Ypred == Ynew) / length(Ynew) * 100;

end
