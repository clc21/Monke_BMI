clc; close all; clear all;

%% data

% Load training data
load('monkeydata_training.mat');

% Number of angles and neurons
[n,k] = size(trial); % n = num_trials
angles = 1:k;
num_neurons = 1:size(trial(1,1).spikes, 1); % 98 neurons


%% k-fold cross validation (10-fold)
% X_train = feature (ie. mean spike rate)
% y_train = labels (ie. angles)


% Initialize feature matrix & label vector
X_train = []; % feature vector
y_train = []; % label

% Extract feature (mean spike rate) for each trial
for angle = angles
    for t = 1: n
        [meanSpikeRate, ~] = extract(trial, angle);
        meanSpikeTrial = mean(meanSpikeRate, 2)'; % average to get a single feature vector
        X_train = [X_train; meanSpikeTrial]; % feature vector
        y_train = [y_train; angle];          % labels
    end
end

% to select the optimal k
k_val = 1:10;
cv_errors = zeros(size(k_val));

for i = 1:length(k_val)
    knnModel = fitcknn(X_train, y_train, 'NumNeighbors', k_val(i), 'CrossVal', 'on', 'KFold', 10);
    % y_pred = predict(model, X_test);
    cv_errors(i) = kfoldLoss(knnModel); % compute cross-validation error
end

% Find the best k (minimum error)
[~, best_k_idx] = min(cv_errors);
best_k = k_val(best_k_idx);

% Train final knn model with best k
best_knnModel = fitcknn(X_train, y_train, 'NumNeighbors', best_k);
y_pred = predict(best_knnModel, X_train);
    
%% Test test

% Randomly Select Angles & Neurons
rng('shuffle');  % Ensure randomness on every run
rand_angles = randperm(8, 4);  % Select 4 random angles
rand_neurons = randperm(98, 50); % Select 50 random neurons

fprintf('Selected Angles: %s\n', mat2str(rand_angles));
fprintf('Selected Neurons: %s\n', mat2str(rand_neurons));

% Extract Training Data
X_train = [];
y_train = [];

for angle = rand_angles  % Loop through selected angles
    [meanSpikeRate, ~] = extract(trial, angle);  % Extract feature matrix
    
    for t = 1:size(meanSpikeRate, 2)  % Loop over trials
        meanSpikeTrial = meanSpikeRate(rand_neurons, t)'; % Select random neurons
        
        % Append data
        X_train = [X_train; meanSpikeTrial]; % Features
        y_train = [y_train; angle];          % Labels
    end
end

% Select Optimal k using Cross-Validation
k_values = 1:10;
cv_errors = zeros(size(k_values));

for i = 1:length(k_values)
    knnModel = fitcknn(X_train, y_train, 'NumNeighbors', k_values(i), 'CrossVal', 'on', 'KFold', 10);
    cv_errors(i) = kfoldLoss(knnModel);  % Compute error
end

[~, best_k_idx] = min(cv_errors);
best_k = k_values(best_k_idx);

% Train Final k-NN Model
best_knnModel = fitcknn(X_train, y_train, 'NumNeighbors', best_k);

X_test = [];
y_test = [];

for angle = rand_angles  % Test on same selected angles
    [meanSpikeRate, ~] = extract(trial, angle);
    
    for t = 1:size(meanSpikeRate, 2)
        meanSpikeTrial = meanSpikeRate(rand_neurons, t)'; % Select random neurons
        
        % Append test data
        X_test = [X_test; meanSpikeTrial];
        y_test = [y_test; angle];
    end
end

% Predict Angles
y_pred = predict(best_knnModel, X_test);

% Display Predicted vs Actual Angles
disp('Predicted Angles vs. Actual Angles:');
disp(table(y_pred, y_test));

% Calculate Accuracy
accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

    
    