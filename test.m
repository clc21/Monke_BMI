% load("monkeydata_training.mat")

% Kalman = KalmanFilterRegression(0.1);
%%
angle=1;
for t=1:100
    [spikeRate,handKinematics,time_bins] = extractWindows(trial,t,angle,winStp=20,winSz=20);
    Kalman.setInitialPos(handKinematics(1:2,1));
    Kalman.fit(spikeRate,handKinematics);
    Kalman.predict(spikeRate,handKinematics);
    disp(['angle grp:',num2str(angle),'   trial:',num2str(t)]);
    Kalman.plotValues(true);
    

    pause(0.1);
end
Kalman.clearRMSe();
% figure(2);
% plot(Kalman.RMSe);
% title('RMSe per trial')



