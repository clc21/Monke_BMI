%% Training
clc; close all; clear all;

data = load('monkeydata_training.mat');
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%% POSITION ESTIMATOR FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters)

    persistent predictedAngle
    
   % On t = 320, bin test data in 20ms to get 98 x 16. For now,
    % use manual method
    n_neurons = size(past_current_trial.spikes, 1);  % Number of neurons
    n_time = size(past_current_trial.spikes, 2);     % Total time points
    n_wind = floor((n_time - 20) / 20) + 1; % Number of windows
    
    % Initialize arrays
    testspikeRate = zeros(n_neurons, n_wind);
    time_bins = zeros(1, n_wind);

    % Sliding window loop
    for w = 1:n_wind
        % Define window range
        t_start = (w - 1) * 20 + 1; % 20 is args.winStp
        t_end = t_start + 20 - 1;
    
        % Compute spike rate
        testspikeRate(:, w) = sum(past_current_trial.spikes(:, t_start:t_end), 2) / 20; 
        % Try square root to imrpve accuracy >  helps reduce the effect of high-firing neurons and stabilizes variance.
        testspikeRate(:, w) = sqrt(testspikeRate(:, w));
        % Store time bin center
        time_bins(w) = round((t_start + t_end) / 2);
    end
    
    % Apply Gaussian smoothing. Use manual method for  now,
    sigma = 50;  % Standard deviation for Gaussian smoothing (in ms)
    winSz = 10;  % Window size in ms
    % Create Gaussian kernel
    kernelSize = ceil(3 * sigma / winSz) * 2 + 1;  % Ensure it's odd
    t_kernel = linspace(-kernelSize / 2, kernelSize / 2, kernelSize);
    gaussKernel = exp(-t_kernel.^2 / (2 * sigma^2));
    gaussKernel = gaussKernel / sum(gaussKernel);  % Normalize
    % Apply convolution along the time dimension (2nd axis)
    smoothedtestspikeRate = conv2(testspikeRate, gaussKernel, 'same');
    
    % Flatten spike rate matrix to column matrix to prepare for
    % projection
    test_flat = smoothedtestspikeRate(:);
    
    % increment count until reaches maximum (= 13). Afterwards, just use the same predicted angle until 'direc' loops over new angle.
    % Think won't cause inaccuracy in prediction as long as prior prediction is correct

    if n_time <= 560 
        % Center test spike rate with corresponding trained meanVec
        count = (n_time - 300)/20;
        centered_test = test_flat - modelParameters.meanVec_all{count};
        
        % project test spike rate into corresponding projection matrix
        W_test = modelParameters.projMat_all{count}' * centered_test;  
        W_train = modelParameters.W_all{count};

        % Predict reaching angle
        knnModel = modelParameters.bestKNNModels{count};
        predictedAngle = predict(knnModel, W_test'); 
        predictedAngle = double (predictedAngle);
    else
        predictedAngle = predictedAngle;
        % new predicted angle = predicted angle from prior t loop
    end
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
    % UPDATED TO LATEST IMRPOVEMENT
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
    

filtered_data = filterTrial(trainingData, 'min');
    
% Get spike rate matrices with Gaussian smoothing
[~, ~, spikeRateMat] = getSmoothedSpikeRate(filtered_data);

% Loop over rate matrix to find the shortest time bin
min_cols_per_column = zeros(1, size(spikeRateMat, 2));

for col = 1:size(spikeRateMat, 2)
    col_lengths = zeros(1, 1);  

    for row = 1:size(spikeRateMat, 1)
        matrix = spikeRateMat{row, col};            % Access the 98xN matrix
        col_lengths(row) = size(matrix, 2);  % Get the number of time steps
    end

    min_cols_per_column(col) = min(col_lengths);  
end

% set time bin to extract rate
minTrimmer = 320/20; % start trimming training data at 320/20 bin to match causal test time starting at 320 ms
maxTrimmer = min(min_cols_per_column); % Find min in this column

% Initialize cell arrays to store projections and related parameters
projMat_all = cell(maxTrimmer-minTrimmer + 1, 1);
meanVec_all = cell(maxTrimmer-minTrimmer + 1, 1);
W_all = cell(maxTrimmer-minTrimmer + 1, 1);
count = 1;

% Loop through each trimmer
for trimmer = minTrimmer:maxTrimmer
    % Prepare firingData: flatten spikeRate matrices up to `trimmer` bins
    [nTrials, nAngles] = size(spikeRateMat);
    nNeurons = size(spikeRateMat{1,1}, 1);  % e.g., 98 neurons

    X_pca = zeros(nNeurons * trimmer, nTrials * nAngles);

    for angle = 1:nAngles
        for trial = 1:nTrials
            % Get only the first `trimmer` columns of spike rate
            spikeRate_trimmed = spikeRateMat{trial, angle}(:, 1:trimmer);
            X_pca(:, (angle - 1) * nTrials + trial) = spikeRate_trimmed(:);
        end
    end
    
    % Apply PCA
    [eigVectors, afterPCA, meanVec] = applyPCA(X_pca);

    % Apply LDA
    [W, projMat] = applyLDA(afterPCA, X_pca, filtered_data);

    % Store results
    projMat_all{count} = projMat;
    meanVec_all{count} = meanVec;
    W_all{count} = W;
    count = count+1;
    % Xtrain = W';
    % gscatter(Xtrain(:,1), Xtrain(:,2));
end

% best_k refer to overleaf report = 5
best_k = 5;
% Labels for 8 directions, 50 trials each
labels = repelem(1:8, 50)';
labels = categorical(labels);  % Convert to categorical if needed

bestKNNModels = cell(length(W_all), 1);

for i = 1:length(W_all)
    X_train = W_all{i}';  % Now it's 400 × d
    y_train = labels;

    % Train final kNN model using best_k
    bestKNNModels{i} = fitcknn(X_train, y_train, 'NumNeighbors', best_k);
end


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
    modelParameters.W_all = W_all;                        % LDA projection matrices
    modelParameters.projMat_all = projMat_all;            % Projected features after LDA
    modelParameters.meanVec_all = meanVec_all;            % PCA mean vectors
    modelParameters.bestKNNModels = bestKNNModels;        % Trained KNN models for each time step
    fprintf('Model training completed successfully.\n');
end

%%%%%%%%%%%%%%%%%%%% ALGORITHM FUNCTIONS %%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%% PREPROCESSING FUNCTION %%%%%%%%%%%%%%%

function [filtered_data] = filterTrial(data, matchLengthMethod)
    arguments
        data struct
        matchLengthMethod char {mustBeMember(matchLengthMethod, {'max', 'min'})} = 'max'
    end
    
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

    % Might be unused due to abandonment
    % Determine all valid lengths
    for trial_idx = 1:n_trials
        for angle_idx = 1:n_angles
            total_time = size(data.trial(trial_idx, angle_idx).spikes, 2);
            start_idx = 1; 
            end_idx = total_time;
            
            if end_idx > start_idx
                lengths_count = lengths_count + 1;
                trial_lengths(lengths_count) = end_idx - start_idx + 1;
            end
        end
    end
    
    % Trim any unused elements in the array
    trial_lengths = trial_lengths(1:lengths_count);

    % Abandon length matching for time being
    % % Get target length in one operation
    % if strcmp(matchLengthMethod, 'max')
    %     target_length = max(trial_lengths);
    % else
    %     target_length = min(trial_lengths);
    % end
    target_length = trial_lengths;

    % Process all trials
    for trial_idx = 1:n_trials
        for angle_idx = 1:n_angles
            current_trial = data.trial(trial_idx, angle_idx);
            total_time = size(current_trial.spikes, 2);
            start_idx = 1; % Take full length for now
            end_idx = total_time; % Account the last 100 ms
            
            if end_idx <= start_idx
                continue;
            end
            
            % Process spike data
            if isfield(current_trial, 'spikes')
                extracted_spikes = current_trial.spikes(:, start_idx:end_idx);
                % Abandon length matching for now
                % [num_channels, current_length] = size(extracted_spikes);
                % 
                % if current_length < target_length()
                %     % Pad with zeros - preallocate for speed
                %     adjusted_spikes = zeros(num_channels, target_length);
                %     adjusted_spikes(:, 1:current_length) = extracted_spikes;
                % else
                %     % Truncate to target length
                %     adjusted_spikes = extracted_spikes(:, 1:target_length);
                % end
                % 
                filtered_data(trial_idx, angle_idx).spikes = extracted_spikes;
            end
            
            % Process handPos data
            if isfield(current_trial, 'handPos')
                extracted_handPos = current_trial.handPos(:, start_idx:end_idx);
                [num_channels, current_length] = size(extracted_handPos);
                % Abandon length matching for now
                % if current_length < target_length
                %     % Pad with zeros - preallocate for speed
                %     adjusted_handPos = zeros(num_channels, target_length);
                %     adjusted_handPos(:, 1:current_length) = extracted_handPos;
                % else
                %     % Truncate to target length
                %     adjusted_handPos = extracted_handPos(:, 1:target_length);
                % end
                
                filtered_data(trial_idx, angle_idx).handPos = extracted_handPos;
            end
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

    % Compute spike rate
    spikeRate(:, w) = sum(spike_train(:, t_start:t_end), 2) / args.winSz; 
    % Try square root to improve accuracy
    spikeRate(:, w) = sqrt(spikeRate(:, w));
    % Store time bin center
    time_bins(w) = round((t_start + t_end) / 2);
end
end


%%%%%%%%%%%%%% PCA FUNCTION %%%%%%%%%%%%%%%
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

    % % Find out the variance for the sake of report. Not needed in predicting angle
    % varianceExplained = eigValues / sum(eigValues) * 100;
    % cumulativeVariance = cumsum(varianceExplained);
    % 
    % % Plotting cumulative variance 
    % figure;
    % plot(cumulativeVariance, 'o-', 'LineWidth', 2);
    % xlabel('Number of Principal Components');
    % ylabel('Cumulative Variance Explained (%)');
    % title('PCA: Cumulative Variance Explained');
    % yline(95, '--r', '95% Threshold');
    % yline(99, '--g', '99% Threshold');
    % grid on;

end


%%%%%%%%%%%%%% LDA FUNCTION %%%%%%%%%%%%%%%
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
    
    pcaDim = 296; % Keep the first 296 components from PCA (==> % variance)
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


%%%%%%%%%%%%%% KNN REGRESSION CLASS %%%%%%%%%%%%%%%

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

function [spikeRate,handKinematics,time_bins] = extractFeatures(trial,args)
%extractWindows Extract the spike rate, hand Pos&Vel&Acc in time windows
%   trial
arguments
    trial struct
    args.trialNumber double = 0
    args.angle double = 0
    args.isStruct logical = false
    args.winSz double = 10  % Window size in ms
    args.winStp double = 10 % Window step in ms
    args.sigma double = 50  % Standard deviation for Gaussian smoothing (in ms)
end

if args.isStruct
    spike_train = trial(args.trialNumber,args.angle).spikes;
    n_time = size(spike_train,2);
    try
        pos_array = trial(args.trialNumber,args.angle).handPos;
    catch
        pos_array = zeros(3,n_time);
    end
else
    spike_train = trial.spikes;
    n_time = size(spike_train,2);
    try
        pos_array = trial.handPos;
    catch
        pos_array = zeros(3,n_time);
    end
end

dt = 1e-3;                            % sampling period
n_neurons = size(spike_train,1);      % number of neurons
n_pos = size(pos_array,1);            % number of directions
time_range = 1:size(spike_train,2);  % slice of interest
n_wind = floor((n_time - args.winSz) / args.winStp) + 1;

% velocity is derivative of displacement
pos_prev = zeros(n_pos,n_time);
pos_prev(:,1) = pos_array(:,1);
pos_prev(:,2:end) = pos_array(:,1:end-1);
vel_array = (pos_array-pos_prev)/dt;

% acceleration is derivative of velocity\
vel_prev = zeros(n_pos,n_time);
vel_prev(:,1) = vel_array(:,1);
vel_prev(:,2:end) = vel_array(:,1:end-1);
acc_array = (vel_array-vel_prev)/dt;

% initialise arrays
spikeRate = zeros(n_neurons, n_wind);
handPos = zeros(n_pos,n_wind);      % average displacement per window
handVel = zeros(n_pos,n_wind);      % average velocity per window
handAcc = zeros(n_pos,n_wind);      % average acceleration per window
time_bins = zeros(1, n_wind);

% Sliding window loop
for w = 1:n_wind
    % Define window range (within 300:600)
    t_start = (w-1) * args.winStp + 1;
    t_end = t_start + args.winSz - 1;

    % Compute spike rate and kinematics
    spikeRate(:, w) = sum(spike_train(:, t_start:t_end), 2)/ args.winSz * 1000; % Convert to Hz
    handPos(:, w) = mean(pos_array(:, t_start:t_end),2);
    handVel(:, w) = mean(vel_array(:, t_start:t_end),2);
    handAcc(:, w) = mean(acc_array(:, t_start:t_end),2);

    % Store time bin center (relative to the time range)
    time_bins(w) = time_range(round((t_start + t_end) / 2));
end

% Apply Gaussian smoothing to spike rates
spikeRate = applyGaussianFilter(spikeRate, args.sigma, args.winSz);

handKinematics = cat(1,handPos(1:2,:),handVel(1:2,:),handAcc(1:2,:));

end

%%%%%%%%%%%%%% KALMAN REGRESSION CLASS %%%%%%%%%%%%%%%
classdef KalmanFilterRegression < handle
    % Class for the Kalman Filter Decoder
    properties (Dependent)
        model
    end
    properties
        n_trials
        Kx
        Ky
        RMSE_history
        RMSe
        Xpred
        Xhist 
    end
    properties (Access = private)
        A   % state transition matrix (nxn)
        H   % state-to-measurement/observation matrix (mxn)
        W   % process noise covariance   
        Q   % measurement covariance
        P   % estimate covariance
        K   % Kalman gain
        X   % state estimate
        alpha double    % learning rate for system identification
        sysID logical
        n_states
        delayStp
        Xtrue
        measurementBuffer
    end

    methods
        function obj = KalmanFilterRegression(args)
            arguments
                args.alpha double = 0.1;
                args.n_neurons double = 98;
                args.binSize double = 10;  %in ms
                args.delaySteps double = 0;
            end
            n_neurons = args.n_neurons;
            n_states = 6;  % disp,vel,accel
            obj.n_states = n_states;
            obj.sysID = false;

            % Initialize delay buffer
            obj.delayStp = args.delaySteps;
            obj.measurementBuffer = [];

            % Initialize history tracking
            obj.Kx = [];
            obj.Ky = [];
            obj.RMSE_history = [];
            obj.RMSe = [];

             % Initialize state transition matrix with velocity and acceleration model
            dt = args.binSize/1000; % convert bin size to seconds
            dt2 = 0.5*dt^2;
            obj.A = [1 0 dt 0  dt2 0;
                     0 1 0  dt 0   dt2;
                     0 0 1  0  dt  0;
                     0 0 0  1  0   dt;
                     0 0 0  0  1   0;
                     0 0 0  0  0   1];               
            obj.H = zeros(n_neurons,n_states);  
            obj.W = eye(n_states) * 0.01; 
            obj.Q = eye(n_neurons) * 0.1;
            obj.P = eye(n_states) * 0.1;
            obj.X = zeros(n_states,1);      % Initial state estimate
            obj.K = zeros(n_states,n_neurons);

            obj.alpha = args.alpha;
        end

        function value = get.model(obj)
            mod.A = obj.A;               
            mod.H = obj.H;  
            mod.W = obj.W; 
            mod.Q = obj.Q; 
            mod.P = obj.P;
            mod.X = obj.X;      % Initial state estimate
            mod.K = obj.K;
            value = mod;
        end

        function obj = fit(obj, Z_train, X_train)
            % Train Kalman Filter Decoder [row,col]
            % Z_train: [n_neurons,n_samples] - Neural data
            % X_train: [n_outputs,n_samples] - Outputs to predict
            % if ~obj.sysID
            %     obj.sysID = true;
            %     a = 1;
            % else
            %     a = obj.alpha;
            % end

            a = obj.alpha;

            M = size(X_train,2);  % n_samples

            X0 = X_train;
            Z0 = Z_train;
            X1 = X_train(:,1:M-1);
            X2 = X_train(:,2:M);

            calcA = X2 * X1' / (X1 * X1');
            obj.A = (1-a)*obj.A + a * calcA;

            calcH = Z0 * X0' / (X0 * X0');
            obj.H = (1-a)*obj.H + a * calcH;

            calcW = (X2 - obj.A * X1) * (X2 - obj.A * X1)' / (M-1);
            obj.W = (1-a)*obj.W + a * calcW;

            calcQ = ((Z0 - obj.H * X0) * (Z0 - obj.H * X0)' )/ M;
            obj.Q = (1-a)*obj.Q + a * calcQ;
        end

        function obj = predict(obj,Z_train,X_train)
            arguments
                obj
                Z_train
                X_train = false; 

            end
            obj.clearHistory();

            k_steps = size(Z_train, 2);

            for t = 1:k_steps
                if X_train
                    obj.updateLoop(Z_train(:, t), X_train(:, t));  
                else
                    obj.updateLoop(Z_train(:, t),X_train);  
                end
            end
            obj.RMSe(end+1) = mean(obj.RMSE_history);
        end

        function obj = updateLoop(obj, Z, X_true)
            % Recursive update step
            % Z: Neural activity (spike counts) at current timestep
            % X_true: True state (X and Y pos)
            obj.measurementBuffer = [obj.measurementBuffer, Z];
            regInv = eye(size(obj.Q)) * 1e-6;

            % Check if enough measurements have been buffered
            if size(obj.measurementBuffer,2) > obj.delayStp
                % Retrieve delayed measurement
                Z_delayed = obj.measurementBuffer(:, 1);
                
                % Remove used measurement from buffer
                obj.measurementBuffer(:, 1) = [];
            else
                % If buffer is not full, skip update
                Z_delayed = zeros(size(Z));
            end

            % Time update %
            % ----------- %
            X_pred = obj.A * obj.X;                  % Predict state
            P_pred = obj.A * obj.P * obj.A' + obj.W; % Predict error covariance
        
            % Measurement update %
            % ------------------ %
            % Kalman gain
            obj.K = P_pred * obj.H' / (obj.H * P_pred * obj.H' + obj.Q + regInv); 
            % Update estimate uncertainty
            obj.P = (eye(size(obj.P)) - obj.K * obj.H) * P_pred;
            % Update state estimate
            obj.X = X_pred + obj.K * (Z_delayed - obj.H * X_pred);

            % Save the Position states only
            obj.Xpred(:,end+1) = obj.X(1:2);

            if X_true 
                obj.Xtrue(:,end+1) = X_true(1:2,:);

                % Compute RMSE for position only
                error = X_true(1:2,:) - obj.X(1:2,:);
                obj.RMSE_history(end+1) = sqrt(mean(error.^2));
            end
        end
        function plotValues(obj,RMSEperTrial)
            arguments
                obj
                RMSEperTrial logical = true;
            end
            time = size(obj.Xpred,2);
            figure(1);
            clf(1);
            subplot(1,2,1);
            hold on
            plot(1:time,obj.Xpred,LineWidth=1.5,Color='b');
            hold on
            plot(1:time,obj.Xtrue,'LineStyle','--',LineWidth=1.0,Color='r');
            xlabel('time bins / window')
            ylabel('XY positions / cm')
            legend({'Xpred','Ypred','Xreal','Yreal'},Location="northwest");

            title('Real vs Predicted values');

            subplot(1,2,2);
            hold on
            if RMSEperTrial
                rms_t = 1:length(obj.RMSe);
                plot(rms_t,obj.RMSe,LineWidth=1.0,Color='g');
                title('RMSe per trial');
                xlabel('Trial number');
            else
                plot(1:time,obj.RMSE_history,'LineStyle','-',LineWidth=1.0,Color='g');
                title('RMSe in trial');
                xlabel('time bins / window')
            end
            ylabel('RMSe /cm')

        end
        function obj = clearRMSe(obj)
            obj.RMSe = [];
        end
        function obj = clearHistory(obj)
            obj.Xpred = [];
            obj.Xtrue = [];
            obj.RMSE_history = [];
        end
        function [Xpred,RMSe] = getTrialHistory(obj)
            Xpred = obj.Xpred;
            RMSe = obj.RMSE_history;
        end
        function obj = setInitialPos(obj,XYpos)
            obj.X = cat(1,XYpos,zeros(obj.n_states-2,1));
        end
        function obj = addModel(obj,model)
            arguments
                obj
                model (1,1) struct
            end
            obj.A = model.A;               
            obj.H = model.H;  
            obj.W = model.W; 
            obj.Q = model.Q; 
            obj.P = model.P;
            obj.K = model.K;
        end
        function [X,Y] = getHandPos(obj)
            X = obj.X(1,:);
            Y = obj.X(2,:);
        end
    end
end
