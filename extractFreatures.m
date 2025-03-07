function [spikeCount,spikeRateMean,spikeRateVar,isiMean, isiVar, isiMed] = extractFreatures(spikeMatrix, windowSz, fs)
%EXTRACTFEATURES Summary of this function goes here
%   rastor M - M(cols) timestep and N(row) neurons
%   spikeRate = spikes per window / window length  (normalised spike count)
%   ISImean and var = Inter-Spike Interval (ISI) Statistics
arguments
    spikeMatrix
    windowSz double = 10  % in ms
    fs double = 1e3
end
[spikeCount,spikeRateMean,spikeRateVar] = calc_spike_rate(spikeMatrix,windowSz,fs);
[isiMean, isiVar, isiMed] = calculate_isi_stats(spikeMatrix);
% psd = calc_psd();
end

function [spikeCount,spikeRateMean,spikeRateVar] = calc_spike_rate(spikeMatrix, windowSize, fs)
    % Function to calculate spike rate from a binary spike train
    % 1
    % spikeTrain: Binary vector (1 = spike, 0 = no spike)
    % windowSize: Size of the window
    % fs: Sampling frequency (Hz)
    %
    % Returns:
    % spikeRate: Matrix of spike rates for each window (Hz)
    [numNeurons, numTimeSteps] = size(spikeMatrix);
    
    % Number of windows
    numWindows = floor(numTimeSteps / windowSize);
    
    % Initialize spike rate matrix
    spikeCount = zeros(numNeurons, numWindows);

    realWindSz = windowSize / fs;
    
    % Compute spike rate for each neuron
    for w = 1:numWindows
        startIdx = (w - 1) * windowSize + 1;
        endIdx = startIdx + windowSize - 1;
        spikeCount(:, w) = sum(spikeMatrix(:, startIdx:endIdx),2);
    end
    spikeRate = spikeCount/realWindSz;
    spikeRateMean = mean(spikeRate,2); % Spikes per second (Hz)
    spikeRateVar = var(spikeRate,0,2);
end


function [isiMean, isiVar, isiMedian] = calculate_isi_stats(spikeMatrix)
    % Function to compute ISI statistics (mean, variance, median) per neuron
    %
    % spikeMatrix: Binary matrix (neurons x time)
    % fs: Sampling frequency (Hz)
    %
    % Returns:
    % isiMean: Mean ISI per neuron (in seconds)
    % isiVar: Variance of ISI per neuron (in seconds^2)
    % isiMedian: Median ISI per neuron (in seconds)

    [numNeurons, ~] = size(spikeMatrix);
    
    % Initialize output vectors
    isiMean = nan(numNeurons, 1);
    isiVar = nan(numNeurons, 1);
    isiMedian = nan(numNeurons, 1);
    
    for n = 1:numNeurons
        % Find spike times (indices where spike occurs)
        spikeTimes = find(spikeMatrix(n, :) == 1);
        
        if length(spikeTimes) > 1
            % Compute ISI (time difference between consecutive spikes)
            isi = diff(spikeTimes) * 1e-3; % Convert to seconds
            
            % Compute ISI statistics
            isiMean(n) = mean(isi);
            isiVar(n) = var(isi);
            isiMedian(n) = median(isi);
        end
    end
end


