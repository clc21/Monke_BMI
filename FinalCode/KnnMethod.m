classdef KnnMethod < handle
    % KnnMethod - A class for KNN-based angle classification in BMI
    
    properties
        % Parameters
        featureType char = 'pca'  % Type of feature extraction method
        winSz double = 10         % Window size in ms
        winStp double = 10        % Window step in ms
        sigma double = 30         % Standard deviation for Gaussian smoothing
        
        % Training data
        X_train                   % Feature matrix for training
        y_train                   % Labels for training
        X_test                    % Feature matrix for testing
        y_test                    % Labels for testing
        
        % Model parameters
        best_k                    % Best k value for KNN
        pcaComponents             % PCA components
        pcaReducedFeatures        % Reduced features after PCA
        ldaProjection             % LDA projection matrix
        
        % Results
        accuracy                  % Accuracy of the model
        
        % Data
        filteredData              % Filtered dataset
        spikeRateData             % Spike rate data
        pca_data                  % Data after PCA
    end
    
    methods
        function obj = KnnMethod(args)
            % Constructor for KnnMethod class
            arguments
                args.featureType char = 'pca'
                args.winSz double = 10
                args.winStp double = 10
                args.sigma double = 30
            end
            
            obj.featureType = args.featureType;
            obj.winSz = args.winSz;
            obj.winStp = args.winStp;
            obj.sigma = args.sigma;
        end
        
        function [Ypred, best_k, accuracy, best_knnModel] = runPipeline(obj, data, matchLengthMethod)
            % Run the entire KNN pipeline
            arguments
                obj
                data struct
                matchLengthMethod char = 'min'
            end
            
            % Filter the data
            obj.filterData(data, matchLengthMethod);
            
            % Extract PCA features
            [eigValues, pcaFeatures] = obj.getPCA(obj.filteredData);
            
            % Always apply LDA (to match main.m)
            [W, projMat] = obj.applyLDA(pcaFeatures, obj.pca_data, obj.filteredData);
            [X, y, Xnew, Ynew, ~] = obj.extractKnnFeature(W, obj.filteredData);
            
            % Store training data
            obj.X_train = X;
            obj.y_train = y;
            obj.X_test = Xnew;
            obj.y_test = Ynew;
            
            % Run cross-validation
            [Ypred, accuracy, k_optimal, best_knnModel] = obj.knn_crossVal(X, y, Xnew, Ynew);
            
            % Store results
            obj.accuracy = accuracy;
            obj.best_k = k_optimal;
            best_k = k_optimal;
        end
                
        function obj = filterData(obj, data, matchLengthMethod)
            % Filter and preprocess the data
            arguments
                obj
                data struct
                matchLengthMethod char = 'min'
            end
            
            % Check if data is already a structure with trial field
            if isfield(data, 'trial')
                input_data = data;
            else
                % Create a temporary structure if data doesn't have trial field
                input_data = struct('trial', data);
            end
            
            % Get dimensions of the input data
            if isfield(input_data, 'trial')
                [n_trials, n_angles] = size(input_data.trial);
            else
                [n_trials, n_angles] = size(input_data);
            end
            
            % Initialize filtered data
            filtered_data = input_data;
            
            % Calculate trial lengths
            trial_lengths = zeros(n_trials * n_angles, 1);
            lengths_count = 0;
            
            for trial_idx = 1:n_trials
                for angle_idx = 1:n_angles
                    % Get spike data length
                    if isfield(input_data, 'trial')
                        total_time = size(input_data.trial(trial_idx, angle_idx).spikes, 2);
                    else
                        total_time = size(input_data(trial_idx, angle_idx).spikes, 2);
                    end
                    
                    % Define start and end indices (300ms before, 100ms after)
                    start_idx = 301;
                    end_idx = total_time - 100;
                    
                    if end_idx > start_idx
                        lengths_count = lengths_count + 1;
                        trial_lengths(lengths_count) = end_idx - start_idx + 1;
                    end
                end
            end
            
            % Trim unused elements
            trial_lengths = trial_lengths(1:lengths_count);
            
            % Determine target length
            if strcmp(matchLengthMethod, 'max')
                target_length = max(trial_lengths);
            else % 'min'
                target_length = min(trial_lengths);
            end
            
            % Process trials
            for trial_idx = 1:n_trials
                for angle_idx = 1:n_angles
                    % Get current trial
                    if isfield(input_data, 'trial')
                        current_trial = input_data.trial(trial_idx, angle_idx);
                    else
                        current_trial = input_data(trial_idx, angle_idx);
                    end
                    
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
                            % Pad with zeros
                            adjusted_spikes = zeros(num_channels, target_length);
                            adjusted_spikes(:, 1:current_length) = extracted_spikes;
                        else
                            % Truncate to target length
                            adjusted_spikes = extracted_spikes(:, 1:target_length);
                        end
                        
                        if isfield(input_data, 'trial')
                            filtered_data.trial(trial_idx, angle_idx).spikes = adjusted_spikes;
                        else
                            filtered_data(trial_idx, angle_idx).spikes = adjusted_spikes;
                        end
                    end
                    
                    % Process handPos data
                    if isfield(current_trial, 'handPos')
                        extracted_handPos = current_trial.handPos(:, start_idx:end_idx);
                        [num_channels, current_length] = size(extracted_handPos);
                        
                        if current_length < target_length
                            % Pad with zeros
                            adjusted_handPos = zeros(num_channels, target_length);
                            adjusted_handPos(:, 1:current_length) = extracted_handPos;
                        else
                            % Truncate to target length
                            adjusted_handPos = extracted_handPos(:, 1:target_length);
                        end
                        
                        if isfield(input_data, 'trial')
                            filtered_data.trial(trial_idx, angle_idx).handPos = adjusted_handPos;
                        else
                            filtered_data(trial_idx, angle_idx).handPos = adjusted_handPos;
                        end
                    end
                end
            end
            
            % Save filtered data
            if isfield(input_data, 'trial')
                obj.filteredData = filtered_data.trial;
            else
                obj.filteredData = filtered_data;
            end
            
            % Extract spike rates and apply smoothing
            obj.extractSpikeRates();
        end
        
        function extractSpikeRates(obj)
            % Extract spike rates from filtered data
            [n_trials, n_angles] = size(obj.filteredData);
            spikeRateMatrix = cell(n_trials, n_angles);
            
            for angle = 1:n_angles
                for trial = 1:n_trials
                    % Extract windows from spike data
                    [spikeRate, ~] = obj.extractWindows(obj.filteredData, trial, angle);
                    
                    % Apply Gaussian smoothing
                    spikeRate = obj.applyGaussianFilter(spikeRate);
                    
                    % Store spike rate
                    spikeRateMatrix{trial, angle} = spikeRate;
                end
            end
            
            obj.spikeRateData = spikeRateMatrix;
        end
        
        function [spikeRate, time_bins] = extractWindows(obj, data, trialNumber, angle, varargin)
            % Extract spike rate in time windows
            p = inputParser;
            addOptional(p, 'isStruct', true);
            addOptional(p, 'winSz', obj.winSz);
            addOptional(p, 'winStp', obj.winStp);
            parse(p, varargin{:});
            
            args = p.Results;
            
            if args.isStruct
                spike_train = data(trialNumber, angle).spikes;
            else
                spike_train = data.spikes;
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
                spikeRate(:, w) = sum(spike_train(:, t_start:t_end), 2) / args.winSz * 1000;
                
                % Store time bin center
                time_bins(w) = round((t_start + t_end) / 2);
            end
        end
        
        function smoothedSpikeRate = applyGaussianFilter(obj, spikeRate)
            % Apply Gaussian filter to spike rate
            sigma = obj.sigma;
            winSz = obj.winSz;
            
            % Create Gaussian kernel
            kernelSize = ceil(3 * sigma / winSz) * 2 + 1;  % Ensure it's odd
            t = linspace(-kernelSize / 2, kernelSize / 2, kernelSize);
            gaussKernel = exp(-t.^2 / (2 * sigma^2));
            gaussKernel = gaussKernel / sum(gaussKernel);  % Normalize
            
            % Apply convolution along the time dimension (2nd axis)
            smoothedSpikeRate = conv2(spikeRate, gaussKernel, 'same');
        end
        
        function [eigValues, reducedFeatures] = getPCA(obj, data)
            % Extract PCA features
            
            % Check if spike rate data already exists
            if isempty(obj.spikeRateData)
                obj.extractSpikeRates();
            end
            
            % Prepare data for PCA
            obj.pca_data = obj.inputPCA(obj.spikeRateData);
            
            % Apply PCA
            [eigValues, reducedFeatures, ~] = obj.applyPCA(obj.pca_data);
            
            % Store PCA results
            obj.pcaComponents = eigValues;
            obj.pcaReducedFeatures = reducedFeatures;
        end
        
        function [X_pca] = inputPCA(obj, spikeRateData)
            % Format spike rate data for PCA - matching original implementation
            [n, k] = size(spikeRateData);                       % n = num_trials, k = angles
            [numFeatures, timeBins] = size(spikeRateData{1,1}); % Number of features (neurons) and time bins
            X_pca = zeros(numFeatures*timeBins, n*k);           % Preallocate matrix (neurons*time, trials*angles)
            
            for angle = 1:k
                for t = 1:n
                    % Extract spike rate matrix (numNeurons x timebins)
                    spikeRate = spikeRateData{t, angle};
                    
                    % Flatten the matrix into a column vector
                    flattenedVector = spikeRate(:);
                    
                    % Compute the column index in X_pca
                    colIdx = (angle-1)*n + t;
                    
                    % Assign flattened vector to the corresponding column
                    X_pca(:, colIdx) = flattenedVector;
                end
            end
        end
        
        function [eigValues, reducedFeatures, covMatrix] = applyPCA(obj, X)
            % Apply PCA to data
            
            % Normalize data by subtracting mean
            X = X - mean(X, 2);
            
            % Compute covariance matrix
            covMatrix = (X' * X) / size(X, 2);
            
            % Compute eigenvalues and eigenvectors
            [eigVectors, eigValues] = eig(covMatrix);
            
            % Sort eigenvalues in descending order
            [~, sortIdx] = sort(diag(eigValues), 'descend');
            eigVectors = eigVectors(:, sortIdx);
            
            % Get eigenvalues
            eigValues = diag(eigValues);
            eigValues = diag(eigValues(sortIdx));
            
            % Project data onto new basis
            pc = X * eigVectors;
            reducedFeatures = pc./sqrt(sum(pc.^2));
        end
        
        function [W, projMat] = applyLDA(obj, reducedFeatures, X_pca, data)
            % Apply LDA for dimensionality reduction
            
            [n, k] = size(data);
            dataMean = zeros(size(X_pca, 1), k);
            
            % Get between-class and within-class scatter matrices
            for angle = 1:k
                dataMean(:, angle) = mean(X_pca(:, n*(angle-1)+1:angle*n), 2);
            end
            
            Sb = (dataMean - mean(X_pca, 2)) * (dataMean - mean(X_pca, 2))'; % Between class scatter
            x_grand = (X_pca - mean(X_pca, 2)) * (X_pca - mean(X_pca, 2))';  % Grand mean
            Sw = x_grand - Sb; % Within class scatter
            
            pcaDim = 296; 
            Sw = reducedFeatures(:, 1:pcaDim)' * Sw * reducedFeatures(:, 1:pcaDim);
            Sb = reducedFeatures(:, 1:pcaDim)' * Sb * reducedFeatures(:, 1:pcaDim);
            
            % Ensure Sw is invertible
            if rank(Sw) < size(Sw, 1)
                Sw = pinv(Sw);  % Use pseudo-inverse if singular
            else
                Sw = inv(Sw);   % Regular inverse
            end
            
            [eigVecLDA, eigValLDA] = eig(Sw*Sb);
            
            [~, sortIdx] = sort(diag(eigValLDA), 'descend');
            
            % Optimum output
            projMat = reducedFeatures(:, 1:pcaDim) * eigVecLDA(:, sortIdx(1:2));
            % Optimum projection
            W = projMat' * (X_pca - mean(X_pca, 2));
            
            % Store LDA projection
            obj.ldaProjection = projMat;
        end
        
        function [X, y, Xnew, Ynew, labels] = extractKnnFeature(obj, W, data, varargin)
            % Extract features for KNN with fixed replication
            if nargin > 2 && ~isempty(varargin) && ischar(varargin{1})
                featureType = varargin{1};
            else
                featureType = obj.featureType;
            end
            
            % Get data dimensions
            if isstruct(data)
                [n, k] = size(data);
            else
                n = 1;
                k = 8; % Assume 8 angles
            end
            
            trialDir = size(W, 2);
            
            % Create labels - ensure replication factor is an integer
            repFactor = floor(trialDir/k);
            if repFactor < 1
                repFactor = 1;
            end
            
            % Create labels with integer replication factor
            labels = [];
            for i = 1:k
                labels = [labels; repmat(i, repFactor, 1)];
            end
            
            % Adjust if necessary to match trialDir
            if length(labels) < trialDir
                labels = [labels; repmat(k, trialDir - length(labels), 1)];
            elseif length(labels) > trialDir
                labels = labels(1:trialDir);
            end
            
            % Split data for training/testing
            ratioTrain = 0.8; % 80% training, 20% testing
            numTrain = round(ratioTrain * trialDir);
            
            randIdx = randperm(trialDir); % Shuffle indices
            trainIdx = randIdx(1:numTrain);
            testIdx = randIdx(numTrain+1:end);
            
            X = W(:, trainIdx)';
            Xnew = W(:, testIdx)';
            
            % Fix: Ensure y and Ynew are column vectors
            y = labels(trainIdx);
            y = y(:); % Ensure it's a column vector
            
            Ynew = labels(testIdx);
            Ynew = Ynew(:); % Ensure it's a column vector
        end
        
        function [Ypred, accuracy, best_k, best_knnModel] = knn_crossVal(obj, X, y, Xnew, Ynew)
            % Perform K-nearest neighbors with cross-validation
            
            % Fix: Ensure y and Ynew are column vectors
            y = y(:);
            
            if ~isempty(Ynew)
                Ynew = Ynew(:);
            end
            
            % Select the optimal k
            k_val = [1, 3, 5, 7, 9, 11, 13, 15]; % Changed from 1:2:15 to match main.m
            cv_errors = zeros(size(k_val));
            
            for i = 1:length(k_val)
                knnModel = fitcknn(X, y, 'NumNeighbors', k_val(i), 'CrossVal', 'on', 'KFold', 10);
                cv_errors(i) = kfoldLoss(knnModel); % Compute cross-validation error
            end
            
            % Find best k (minimum error)
            [~, best_k_idx] = min(cv_errors);
            best_k = k_val(best_k_idx);
            
            % Train final knn model with best_k
            best_knnModel = fitcknn(X, y, 'NumNeighbors', best_k);
            Ypred = predict(best_knnModel, Xnew);
            
            % Calculate accuracy
            if ~isempty(Ynew)
                accuracy = sum(Ypred == Ynew) / length(Ynew) * 100;
            else
                accuracy = 0;
            end
            
            % Store best k
            obj.best_k = best_k;
        end
        
        % Add a helper method for prediction
        function angle = predictAngle(obj, spikeData)
            % Create a struct to hold the spike data
            tempData = struct();
            tempData.spikes = spikeData;
            
            % Filter and extract features
            obj.filterData(tempData);
            [~, pcaFeatures] = obj.getPCA(tempData);
            
            % Use simple spike count features if we can't get PCA
            if isempty(pcaFeatures) || isempty(obj.X_train) || isempty(obj.y_train)
                angle = 1;
                return;
            end
            
            % Use direct KNN prediction with stored training data
            try
                % Make sure training data is properly formatted
                X_train = obj.X_train;
                y_train = obj.y_train(:);
                
                % Get average spike rate as feature
                avgRate = mean(pcaFeatures, 2)';
                
                % Use KNN to predict
                mdl = fitcknn(X_train, y_train, 'NumNeighbors', obj.best_k);
                predictions = predict(mdl, avgRate);
                
                % Get mode if multiple predictions
                if length(predictions) > 1
                    angle = mode(predictions);
                else
                    angle = predictions;
                end
                
                % Ensure angle is valid
                angle = max(1, min(8, angle));
            catch
                % If error occurs, use default
                angle = 1;
            end
        end
    end
end