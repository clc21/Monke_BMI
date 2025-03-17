function [decodedPosX, decodedPosY] = positionEstimatorTest(past_current_trial, modelParameters)
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
    persistent predictedAngle Kalman decodedPos lastTrialId
    
    % Check if this is a new trial
    isNewTrial = isempty(predictedAngle) || ...
                 isempty(past_current_trial.decodedHandPos) || ...
                 (isfield(past_current_trial, 'trialId') && ~isequal(past_current_trial.trialId, lastTrialId));
    
    % Update trial ID
    if isfield(past_current_trial, 'trialId')
        lastTrialId = past_current_trial.trialId;
    end
    
    % If first call or new trial, initialize
    if isNewTrial
        try
            % Direct method: simple correlation-based approach to predict angle
            spikeCounts = sum(past_current_trial.spikes, 2);
            
            % Default to angle 1
            predictedAngle = 1;
            bestScore = -inf;
            
            % Find which Kalman model best correlates with the spike data
            for angle = 1:8
                model = modelParameters.kalmanModels{angle};
                H = model.H; % Observation matrix
                
                % Use position component of H for correlation
                positionComponent = H(:,1);
                
                % Compute correlation
                c = corrcoef(spikeCounts, positionComponent);
                score = abs(c(1,2)); % Get absolute correlation
                
                % Update if this is better
                if score > bestScore
                    bestScore = score;
                    predictedAngle = angle;
                end
            end
            
            % Ensure valid angle
            predictedAngle = max(1, min(8, predictedAngle));
            
            % Initialize Kalman filter with the right model
            Kalman = KalmanFilterRegression(n_neurons=98, alpha=0.1, binSize=10, delaySteps=0);
            Kalman.setInitialPos(past_current_trial.startHandPos);
            Kalman.addModel(modelParameters.kalmanModels{predictedAngle});
            
            % Initialize decoded position
            decodedPos = past_current_trial.startHandPos;
        catch e
            fprintf('Error during initialization: %s\n', e.message);
            predictedAngle = 1; % Default to first angle if error occurs
            
            % Initialize Kalman filter with default model
            Kalman = KalmanFilterRegression(n_neurons=98, alpha=0.1, binSize=10, delaySteps=0);
            Kalman.setInitialPos(past_current_trial.startHandPos);
            Kalman.addModel(modelParameters.kalmanModels{predictedAngle});
            
            decodedPos = past_current_trial.startHandPos;
        end
    end
    
    % Extract features from current spike data
    spikes = past_current_trial.spikes;
    
    % Process in 10ms bins (as in training)
    windowSize = 10;
    totalTime = size(spikes, 2);
    
    % Use the most recent window
    if totalTime >= windowSize
        startIdx = totalTime - windowSize + 1;
        spikeRate = sum(spikes(:, startIdx:totalTime), 2) / (windowSize/1000); % Convert to Hz
    else
        spikeRate = sum(spikes, 2) / (totalTime/1000); % Use all available data
    end
    
    % Update Kalman filter prediction
    Kalman.predict(spikeRate);
    [decodedPosX, decodedPosY] = Kalman.getHandPos();
    
    % Update decoded position for next call
    decodedPos = [decodedPosX; decodedPosY];
end