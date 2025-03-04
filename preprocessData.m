function [binnedSpikes, binnedHandPos,activeNeuronsIdx] = preprocessData(trial_data,numTrials,numAngles)
    % [numTrials, numAngles]=size(trial_data);
    binnedSpikes=cell(numTrials, numAngles);
    binnedHandPos=cell(numTrials, numAngles);
    activeNeuronsIdx=cell(numTrials,numAngles);
    
    for trialId=1:numTrials
        for angleId=1:numAngles
            % Get current trial data and length
            currentTrial=trial_data(trialId, angleId);
            trialLength=size(currentTrial.spikes, 2);
            
            % Extract movement period (300ms after start to 100ms before end)
            startId=301; % 300ms after start 
            endId=trialLength-100; % 100ms before end
            
            % Extract relevant spike data and hand positions
            spikeData=currentTrial.spikes(:, startId:endId);
            handposData=currentTrial.handPos(:, startId:endId);

            % Removing inactive neurons
            spikePerNeuron=sum(spikeData,2);
            activeNeurons=(spikePerNeuron>0);
            activeNeuronsIdx{trialId,angleId}=find(activeNeurons);
            spikeData=spikeData(activeNeurons,:);
            
            % Number of bins (10ms per bin)
            numBins=floor((endId-startId+1)/10);
            
            % Initialize binned data
            numActiveNeurons=sum(activeNeurons);
            binnedSpikesMatrix=zeros(numActiveNeurons, numBins);
            binnedHandPosMatrix=zeros(3, numBins);
            
            % Bin the data
            for binId=1:numBins
                % Calculate start and end indices for current bin
                binStartId=(binId-1)*10 + 1;
                binEndId=binId*10;
                
                % Sum spikes in each 10ms bin
                if binEndId<=size(spikeData, 2)
                    % Sum spikes in each 10ms bin (only for active neurons)
                    binnedSpikesMatrix(:, binId)=sum(spikeData(:, binStartId:binEndId), 2);
                    
                    % Take average hand position in each 10ms bin
                    binnedHandPosMatrix(:, binId)=mean(handposData(:, binStartId:binEndId), 2);
                end
            end
            
            % Store binned data
            binnedSpikes{trialId, angleId}=binnedSpikesMatrix;
            binnedHandPos{trialId, angleId}=binnedHandPosMatrix;
        end
    end
end
