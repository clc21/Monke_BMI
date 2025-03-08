clc; close all; clear all;

%% Preprocessing

% Load training data
load('monkeydata_training.mat');
bin_size = 1;
[n, k] = size(trial); % n = number of trials, k = number of reaching angles

% Extract spike trains from all trials
% Flatten the spike trains into a feature matrix


%% W/o Feature Extraction (using sum of spike count)

% Initialise feature matrix and angle labels
X_train = []; % Feature
y_train = []; % Angle

% Initialise trajectory training data
pos_train = cell(k, 1);
spikes_train = cell(k, 1);

% Loop through all trials and angles
for i = 1:n
    for j = 1:k
        spikes_ori = trial(i, j).spikes;
        
        % Obtain actual trial duration
        T = size(spikes_ori, 2); % number of time bins
        start_idx = min(301, T);
        end_idx = min(572, T);
        
        spikes_filtered = spikes_ori(:, start_idx:end_idx) ;   % filter out the first 300ms and last 100ms
        spike_counts = sum(spikes_filtered, 2)' ;              % sum spikes over time per neuron
        X_train = [X_train; spike_counts];                     % Store in feature matrix
        y_train = [y_train; k];                                % Store labels
        
        % Store spike trains and reaching angles
        spikes_train{j} = [spikes_train{j}; {spikes_filtered}];
        pos_train{j} = [pos_train{j}, trial(i, j).handPos];
    end
end
        
        

%% PCA

% Reduce dimensionality 
[coeff, score, ~, ~, explained] = pca(X_train);

% Determine components that explain 95% variance
num_components = find(cumsum(explained) >= 95, 1);

% Selected principal components dataset
X_train_pca = score(:, 1:num_components);


%% Angle Classification - SVM
% For linear:
% angle_classifier = fitcsvm(X_train, y_train, 'KernelFunction', 'linear'); % Train linear SVM
% For non-linear:
% angle_classifier = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf');

% Train SVM Classifier for Angle Prediction
svm_classifier = fitcsvm(X_train_pca, y_train, 'KernelFunction', 'linear');

%% Train Linear Regression Models for hand position estimation
pos_models = cell(k, 1);

for i = 1:k
    % Solve linear regression
    pos_models{i} = spikes_train{i} \ pos_train{i}';
end

%% Predict on Test Data

% Initialise
num_test_trials = size(test_data, 1);
predicted_angles = zeros(n_test, 1);
predicted_pos = cell(num_test_trials, 1);

for t = 1:num_test_trials
    % Sum spike counts for classification
    test_spike_counts = sum(test_data(t).spikes, 2)';

    % Transform test data using PCA
    X_test_pca = (test_spike_counts - mean(X_train)) * coeff(:, 1:num_components);

    % Predict reaching angle using SVM
    predicted_angles(t) = predict(svm_classifier, X_test_pca);

    % Predict trajectory using regression model of predicted angle
    chosen_model = pos_models{predicted_angles(t)};
    predicted_positions{t} = chosen_model' * test_data(t).spikes;
end

