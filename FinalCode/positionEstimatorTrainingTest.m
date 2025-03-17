function [modelParameters] = positionEstimatorTrainingTest(trainingData)
    % positionEstimatorTraining - Train models for BMI neural decoding
    %
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
    
    % Use the fixed KNN method
    knn = KnnMethod();
    [~, knn_accuracy, best_k] = knn.runPipeline(trainingData, 'min');
    fprintf('KNN Angle Classification Accuracy: %.2f%% with k=%d\n', knn_accuracy, best_k);

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
    modelParameters.knn = knn;                   % KNN model for angle classification
    modelParameters.kalmanModels = modelParametersKalman; % Kalman models for each angle
    
    fprintf('Model training completed successfully.\n');
end