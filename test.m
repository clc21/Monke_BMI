% load("monkeydata_training.mat")

Kalman = KalmanFilterRegression();
%%
angle=4;
for t=1:100
    [spikeRate,handKinematics,time_bins] = extractFeatures(trial,trialNumber=t,angle=angle,isStruct=true,winSz=10,winStp=10);
    % [spikeRate,handKinematics,time_bins] = extractFeatures(trial(t,angle),isStruct=false,winSz=20,winStp=20);
    Kalman.setInitialPos(handKinematics(1:2,1));
    Kalman.fit(spikeRate,handKinematics);
    Kalman.predict(spikeRate,handKinematics);
    disp(['angle grp:',num2str(angle),'   trial:',num2str(t)]);
    Kalman.plotValues(true);

    pause(0.1);
end

% Kalman.clearRMSe();



