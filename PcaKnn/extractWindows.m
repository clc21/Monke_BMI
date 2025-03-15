function [spikeRate, time_bins] = extractWindows(trial, trialNumber, angle, args)
%extractWindows Extract the spike rate in time windows
%   trial
arguments
    trial struct
    trialNumber double
    angle double
    args.isStruct logical = true
    args.winSz double = 20  % ms
    args.winStp double = 20
end

if args.isStruct
    spike_train = trial(trialNumber, angle).spikes;
else
    spike_train = trial.spikes;
end

n_neurons = size(spike_train, 1);  % Number of neurons
n_time = size(spike_train, 2);     % Total time points
n_wind = floor((n_time - args.winSz) / args.winStp) + 1; % Number of windows

% Initialize arrays
spikeRate = zeros(n_neurons, n_wind);
time_bins = zeros(1, n_wind);

% Sliding window loop
for w = 1:n_wind
    % Define window range
    t_start = (w - 1) * args.winStp + 1;
    t_end = t_start + args.winSz - 1;

    % Compute spike rate (in Hz)
    spikeRate(:, w) = sum(spike_train(:, t_start:t_end), 2) / args.winSz * 1000; 

    % Store time bin center
    time_bins(w) = round((t_start + t_end) / 2);
end
end