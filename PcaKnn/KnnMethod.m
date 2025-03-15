classdef KnnMethod < handle
    % KnnMethod - Streamlined class for decoding reaching angles using KNN
    %
    % This class handles preprocessing neural data, extracting features,
    % and performing KNN classification of reaching angles.
    % It focuses on core functionality with minimal output.
    
    properties
        % Data properties
        filteredData           % Data after length standardization
        pcaParams              % PCA parameters
        pcaFeatures            % PCA-reduced features
        
        % Model properties
        X_train                % Training features
        y_train                % Training labels
        X_test                 % Test features
        y_test                 % Test labels
        
        % Parameters
        windowSize = 20        % Window size in ms
        windowStep = 20        % Window step size in ms
        gaussianSigma = 30     % Sigma for Gaussian smoothing
        trainRatio = 0.8       % Training/testing split ratio
        featureType = 'meanSpikeRate'  % Feature type for KNN
    end
    
    methods
        % Constructor
        function obj = KnnMethod(varargin)
            % Parse input arguments
            p = inputParser;
            p.addParameter('WindowSize', 20, @isnumeric);
            p.addParameter('WindowStep', 20, @isnumeric);
            p.addParameter('GaussianSigma', 30, @isnumeric);
            p.addParameter('TrainRatio', 0.8, @(x) x > 0 && x < 1);
            p.addParameter('FeatureType', 'meanSpikeRate', @(x) ismember(x, {'meanSpikeRate', 'varSpikeRate'}));
            p.parse(varargin{:});
            
            % Set parameters
            obj.windowSize = p.Results.WindowSize;
            obj.windowStep = p.Results.WindowStep;
            obj.gaussianSigma = p.Results.GaussianSigma;
            obj.trainRatio = p.Results.TrainRatio;
            obj.featureType = p.Results.FeatureType;
        end
        
        % Preprocess data
        function filterData(obj, data, lengthMethod)
            if nargin < 3
                lengthMethod = 'min';  % Default to min length
            end
            
            % Filter trials to standardize lengths
            obj.filteredData = obj.filterTrial(data, lengthMethod);
        end
        
        % Apply PCA
        function pcaKNN(obj)
            % Get PCA features
            [obj.pcaParams, obj.pcaFeatures] = obj.getPCA(obj.filteredData);
        end
        
        % Extract features and prepare for training
        function prepareData(obj)
            % Extract features for KNN
            [features, labels] = obj.extractKnnFeature(obj.pcaFeatures, obj.featureType);
            
            % Split data into training and testing sets
            num_samples = size(features, 1);
            shuffle_idx = randperm(num_samples);
            train_size = floor(obj.trainRatio * num_samples);
            
            train_idx = shuffle_idx(1:train_size);
            test_idx = shuffle_idx(train_size+1:end);
            
            obj.X_train = features(train_idx, :);
            obj.y_train = labels(train_idx, :);
            obj.X_test = features(test_idx, :);
            obj.y_test = labels(test_idx, :);
        end
        
        % Train KNN model and evaluate
        function [Ypred, accuracy, best_k] = trainAndEvaluate(obj)
            % Apply KNN
            [Ypred, accuracy, best_k] = obj.knn_crossVal(obj.X_train, obj.y_train, obj.X_test, obj.y_test);
        end
        
        % Run the complete pipeline
        function [Ypred, accuracy, best_k] = runPipeline(obj, data, lengthMethod)
            if nargin < 3
                lengthMethod = 'min';
            end
            
            % Preprocess data
            obj.filterData(data, lengthMethod);
            
            % Apply PCA
            obj.pcaKNN();
            
            % Prepare data for training
            obj.prepareData();
            
            % Train and evaluate
            [Ypred, accuracy, best_k] = obj.trainAndEvaluate();
        end
        
        % Helper functions (converted from the original standalone functions)
        function smoothedSpikeRate = applyGaussianFilter(obj, spikeRate)
            % Create Gaussian kernel
            kernelSize = ceil(3 * obj.gaussianSigma / obj.windowSize) * 2 + 1;  % Ensure it's odd
            t = linspace(-kernelSize / 2, kernelSize / 2, kernelSize);
            gaussKernel = exp(-t.^2 / (2 * obj.gaussianSigma^2));
            gaussKernel = gaussKernel / sum(gaussKernel);  % Normalize

            % Apply convolution along the time dimension (2nd axis)
            smoothedSpikeRate = conv2(spikeRate, gaussKernel, 'same');
        end
        
        function [pcaParams, reducedFeatures] = applyPCA(obj, X, N)
            if nargin < 3
                N = size(X, 1);
            end
            
            % Normalize data
            X = X - mean(X, 1);
            
            % Compute covariance matrix
            covMatrix = cov(X);
            
            % Compute eigenvalues and eigenvectors
            [eigVectors, eigValues] = eig(covMatrix);
            eigValues = diag(eigValues);
            
            % Sort eigenvalues in descending order
            [eigValues, sortIdx] = sort(eigValues, 'descend');
            eigVectors = eigVectors(:, sortIdx);
            
            % Calculate explained variance ratio
            explainedVar = 100 * eigValues / sum(eigValues);

            % Calculate cumulative variance
            cumVar = cumsum(explainedVar);
            
            % Select number of components that explain 95% of variance and ensure
            % we have at least 2 components
            numComponents = find(cumVar >= 95, 1, 'first');
            numComponents = max(numComponents, 2);
            
            % Get projection matrix
            pcaParams.projectionMatrix = eigVectors(:, 1:numComponents);
            pcaParams.explainedVar = explainedVar;  % percentage of explained variance
            pcaParams.numComponents = numComponents;
            pcaParams.meanX = mean(X, 1);
            pcaParams.cumVar = cumVar;
            
            % Project data onto principal components
            reducedFeatures = X * pcaParams.projectionMatrix;
        end
        
        function [X, y] = extractKnnFeature(obj, smoothedSpikeRate, knnFeature)
            [n, k] = size(smoothedSpikeRate);    
            X = [];
            y = [];
            
            for i = 1:n  
                for j = 1:k
                    spikeRate = smoothedSpikeRate{i, j};
                    if strcmp(knnFeature, 'meanSpikeRate')
                        % Use the actual mean per neuron as features
                        knnFeatureVector = mean(spikeRate, 2)';  % Transpose to make a row vector
                    else
                        % Use variance per neuron as features
                        knnFeatureVector = var(spikeRate, 0, 2)';
                    end
                    
                    X = [X; knnFeatureVector];
                    y = [y; j];
                end
            end
        end
        
        function [spikeRate, time_bins] = extractWindows(obj, trial, trialNumber, angle)
            % Extract spike data
            spike_train = trial(trialNumber, angle).spikes;
            
            n_neurons = size(spike_train, 1);  % Number of neurons
            n_time = size(spike_train, 2);     % Total time points
            n_wind = floor((n_time - obj.windowSize) / obj.windowStep) + 1; % Number of windows

            % Initialize arrays
            spikeRate = zeros(n_neurons, n_wind);
            time_bins = zeros(1, n_wind);

            % Sliding window loop
            for w = 1:n_wind
                % Define window range
                t_start = (w - 1) * obj.windowStep + 1;
                t_end = t_start + obj.windowSize - 1;

                % Compute spike rate (in Hz)
                spikeRate(:, w) = sum(spike_train(:, t_start:t_end), 2) / obj.windowSize * 1000; 

                % Store time bin center
                time_bins(w) = round((t_start + t_end) / 2);
            end
        end
        
        function [filtered_data] = filterTrial(obj, data, matchLengthMethod)
            % Check if the data has a 'trial' field
            if ~isfield(data, 'trial')
                error('Input data does not have a "trial" field');
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
                        
                        filtered_data.trial(trial_idx, angle_idx).spikes = adjusted_spikes;
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
                        
                        filtered_data.trial(trial_idx, angle_idx).handPos = adjusted_handPos;
                    end
                end
            end
        end
        
        function [pcaParams_mat, reducedFeatures_mat] = getPCA(obj, data)
            [n, k] = size(data.trial);           % n = num_trials, k = angles
            pcaParams_mat = cell(n, k);          % Store PCA parameters for each angle
            reducedFeatures_mat = cell(n, k);    % Store PCA-reduced features

            for angle = 1:k
                for t = 1:n
                    % Call extractWindows with all required arguments
                    [spikeRate, ~] = obj.extractWindows(data.trial, t, angle);
                    
                    % Apply gaussian smoothing
                    spikeRate = obj.applyGaussianFilter(spikeRate);
                    
                    % Apply PCA - Fixed: call applyPCA instead of pcaKNN
                    [pcaParams, reducedFeatures] = obj.applyPCA(spikeRate);

                    % Store the results
                    pcaParams_mat{t, angle} = pcaParams;
                    reducedFeatures_mat{t, angle} = reducedFeatures;
                end
            end
        end
        
        function [Ypred, accuracy, best_k] = knn_crossVal(obj, X, y, Xnew, Ynew)
            % Select the optimal k (best_k)
            k_val = 1:10;
            cv_errors = zeros(size(k_val));

            for i = 1:length(k_val)
                knnModel = fitcknn(X, y, 'NumNeighbors', k_val(i), 'CrossVal', 'on', 'KFold', 10);
                cv_errors(i) = kfoldLoss(knnModel); % compute cross-validation error
            end
            
            % Find best k (minimum cross-validation error)
            [~, best_idx] = min(cv_errors);
            best_k = k_val(best_idx);
            if mod(best_k, 2) == 0
                best_k = best_k + 1;  % If k is even, add 1 to make it odd
            end
            % if best_k>5
            %     best_k = 5;
            % end 
            
            % Train final model with best k
            final_model = fitcknn(X, y, 'NumNeighbors', best_k);
            
            % Make predictions on new data
            Ypred = predict(final_model, Xnew);
            
            % Calculate accuracy
            accuracy = sum(Ypred == Ynew) / length(Ynew) * 100;
        end
    end
end