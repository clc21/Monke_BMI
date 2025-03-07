% load("monkeydata_training.mat")

Kalman = KalmanFilterRegression(0.2);
%%
angle=1;
for t=1:70
    [meanSpikeRate,~,meanHandPos,~,time_bins] = extract(trial(t,:),angle,'winStp',20,'winSz',20);
    Kalman.fit(meanSpikeRate,meanHandPos(1:2,:));
    Kalman.predict(meanSpikeRate,meanHandPos(1:2,:));
    disp(['angle grp:',num2str(angle),'   trial:',num2str(t)]);
    Kalman.plotValues();


    % subplot(1,2,1)
    % hold on
    % plot(Kalman.Kx');
    % title('Kalman filter coeff for pos X')
    % subplot(1,2,2)
    % hold on
    % plot(Kalman.Ky')
    % title('Kalman filter coeff for pos Y')
    pause(0.5);
end
% figure(2);
% plot(Kalman.RMSe);
% title('RMSe per trial')



