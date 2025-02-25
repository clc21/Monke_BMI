function [rastorMatrix,XYZpos,timeArr] = extract(trial,row,col,args)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
arguments
    trial struct
    row double
    col double
    args.plot logical = false
end

oneTrial = trial(row,col);
rastorMatrix = oneTrial.spikes;
XYZpos = oneTrial.handPos;
timeArr = 1: size(oneTrial.handPos,2);

if args.plot==true
    figure;
    subplot(2,1,1);
    plotSpikeRaster(logical(rastorMatrix),'PlotType','imagesc','TimePerBin',1);
    title(['Rastor plot for trial(',num2str(row),',',num2str(col),')']);
    
    subplot(2,1,2);
    hold on
    plot(timeArr,XYZpos(1,:),LineWidth=2.0,DisplayName="Xpos");
    plot(timeArr,XYZpos(2,:),LineWidth=2.0,DisplayName="Ypos");
    plot(timeArr,XYZpos(3,:),LineWidth=2.0,DisplayName="Zpos");
    legend(Location='northwest');
    title('The XYZ pos from motor neural data')
end

end