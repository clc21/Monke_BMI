function [spikeRate,isiMean, isiVar, isiMed] = extractFreatures(spikeMatrix, windowSz, fs)
%EXTRACTFEATURES Summary of this function goes here
%   rastor M - M(cols) timestep and N(row) neurons
%   spikeRate = spikes per window / window length  (normalised spike count)
%   ISImean and var = Inter-Spike Interval (ISI) Statistics
arguments
    spikeMatrix
    windowSz double = 10  % in ms
    fs double = 1e3
end
spikeRate = calc_spike_rate(spikeMatrix,windowSz,fs);
[isiMean, isiVar, isiMed] = calculate_isi_stats(spikeMatrix);
% psd = calc_psd();
end

function spikeRateMatrix = calc_spike_rate(spikeMatrix, windowSize, fs)
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
    spikeRateMatrix = zeros(numNeurons, numWindows);

    realWindSz = windowSize / fs;
    
    % Compute spike rate for each neuron
    for w = 1:numWindows
        startIdx = (w - 1) * windowSize + 1;
        endIdx = startIdx + windowSize - 1;
        spikeRateMatrix(:, w) = sum(spikeMatrix(:, startIdx:endIdx),2) ./ realWindSz; % Spikes per second (Hz)
    end
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

%% in progress
% function psd = calc_psd(spikeMatrix, fs, freqBands)
%     % Function to compute PSD features from neural spike data
%     % 
%     % spikeMatrix: Binary matrix (neurons x time)
%     % fs: Sampling frequency (Hz)
%     % freqBands: Frequency bands of interest (e.g., [1 4; 4 8; 8 13; 13 30; 30 100] for Delta, Theta, etc.)
%     %
%     % Returns:
%     % psdFeatures: (neurons x number of bands) feature matrix
% 
%     [numNeurons, numTimeSteps] = size(spikeMatrix);
%     numBands = size(freqBands, 1);
%     psdFeatures = zeros(numNeurons, numBands);
% 
%     for n = 1:numNeurons
%         % Convert binary spike train to continuous signal (PSTH-like smoothing)
%         spikeSignal = smoothdata(spikeMatrix(n, :), 'gaussian', round(fs * 0.1)); % 100ms smoothing
% 
%         % Compute Power Spectral Density using Welch’s method
%         [pxx, f] = pwelch(spikeSignal, hamming(256), [], [], fs);
% 
%         % Extract power in predefined frequency bands
%         for b = 1:numBands
%             bandIdx = (f >= freqBands(b, 1)) & (f <= freqBands(b, 2));
%             psdFeatures(n, b) = sum(pxx(bandIdx)); % Total power in band
%         end
%     end
% end

