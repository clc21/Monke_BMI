% load("monkeydata_training.mat")

ix = randperm(length(trial));
trainingData = trial(ix(1:70),:);
testData = trial(ix(71:end),:);

Kalman = KalmanFilterRegression(n_neurons=98,alpha=0.1,binSize=10,delaySteps=0);  % default values
%% Training

modelParameters = repmat(Kalman.model, 1, 8);
 
for angle = 1:8
    Kalman = KalmanFilterRegression(n_neurons=98,alpha=0.1,binSize=10,delaySteps=0);  % default values
    for t=1:70
        [spikeRate,handKinematics,~] = extractFeatures(trainingData,trialNumber=t,angle=angle,isStruct=true,winSz=10,winStp=10);
        % [spikeRate,handKinematics,time_bins] = extractFeatures(trial(t,angle),isStruct=false,winSz=20,winStp=20);
        Kalman.setInitialPos(handKinematics(1:2,1));
        Kalman.fit(spikeRate,handKinematics);
        Kalman.predict(spikeRate,handKinematics);
        disp(['angle grp:',num2str(angle),'   trial:',num2str(t)]);
        % Kalman.plotValues(true);
        % pause(0.1);
    end
    Kalman.clearRMSe();
    pause(0.5);
    modelParameters(angle) = Kalman.model;
end

%% Inference

t = 1;
angle = 1;

[spikeRate,handKinematics,~] = extractFeatures(testData,trialNumber=t,angle=angle,isStruct=true,winSz=10,winStp=10);
% [spikeRate,handKinematics,time_bins] = extractFeatures(trial(t,angle),isStruct=false,winSz=20,winStp=20);

Kalman.setInitialPos(handKinematics(1:2,1));
Kalman.addModel(modelParameters(angle));
Kalman.predict(spikeRate);
[X,Y] = Kalman.getHandPos();
% Kalman.plotValues(false);


