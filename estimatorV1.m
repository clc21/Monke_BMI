% Load data
load("monkeydata_training.mat")
ix = randperm(length(trial));
trainingData = trial(ix(1:70),:);
testData = trial(ix(71:end),:);

Kalman = KalmanFilterRegression(n_neurons=98,alpha=0.1,binSize=10,delaySteps=0);  % default values
knn = KnnMethod();

%% Training
[~, knn_accuracy, best_k] = knn.runPipeline(trainingData, 'min');

fprintf('KNN Angle Classification Accuracy: %.2f%% with k=%d\n', knn_accuracy, best_k);

modelParametersKalman = cell(1, 8);

for angle = 1:8
    Kalman = KalmanFilterRegression(n_neurons=98, alpha=0.1, binSize=10, delaySteps=0);
    
    for t = 1:70
        [spikeRate, handKinematics, ~] = extractFeatures(trainingData, trialNumber=t, angle=angle, isStruct=true, winSz=10, winStp=10);
        
  
        Kalman.setInitialPos(handKinematics(1:2, 1));
        Kalman.fit(spikeRate, handKinematics);
        Kalman.predict(spikeRate, handKinematics);
        
        % disp(['angle grp:', num2str(angle), ' trial:', num2str(t)]);
    end
    
    Kalman.clearRMSe();
    pause(0.5);
    
    modelParametersKalman{angle} = Kalman.model;
end

% Combined model parameters
modelParameters = struct();
modelParameters.knn = knn;  % KNN model for angle classification
modelParameters.kalmanModels = modelParametersKalman;  % Kalman models for each angle


%% Inference

t = 1;
% angle = 1;

tempData = testData(t, 1);

knn.filterData(tempData);
[~, pcaFeatures] = knn.getPCA(tempData);
[features, ~] = knn.extractKnnFeature(pcaFeatures, knn.featureType);

anglePredictions = knn.knn_crossVal(knn.X_train, knn.y_train, features, ones(size(features, 1), 1));
predictedAngle = mode(anglePredictions);  
predictedAngle = max(1, min(predictedAngle, 8));

fprintf('KNN predicted reaching angle: %d\n', predictedAngle);

angle = predictedAngle;

[spikeRate,handKinematics,~] = extractFeatures(testData,trialNumber=t,angle=angle,isStruct=true,winSz=10,winStp=10);
% [spikeRate,handKinematics,time_bins] = extractFeatures(trial(t,angle),isStruct=false,winSz=20,winStp=20);

Kalman.setInitialPos(handKinematics(1:2,1));
Kalman.addModel(modelParameters.kalmanModels{angle});
Kalman.predict(spikeRate);
[X,Y] = Kalman.getHandPos();

fprintf('Predicted position: X=%.2f, Y=%.2f\n', X, Y);
