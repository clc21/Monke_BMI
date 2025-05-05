function [meanSpikeRate,varSpikeRate,meanHandPos,varHandPos,time_bins] = extract(trial,angle,args)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
arguments
    trial struct
    angle double
    args.winSz double = 20  % ms
    args.winStp double = 5
    args.ind_start double = 1
    args.ind_stop double = 550
end

n_trials = size(trial,1);                   % number of trials for one reach angle
n_neurons = size(trial(1,1).spikes,1);      % number of neurons
n_pos = size(trial(1,1).handPos,1);         % number of directions
time_range = args.ind_start:args.ind_stop;  % slice of interest
n_time = length(time_range);
n_wind = floor((n_time - args.winSz) / args.winStp) + 1;

meanSpikeRate = zeros(n_neurons, n_wind);
meanHandPos = zeros(n_pos,n_wind);
varSpikeRate = zeros(n_neurons, n_wind);
varHandPos = zeros(n_pos,n_wind);
time_bins = zeros(1, n_wind);

all_spikes = zeros(n_neurons, n_time, n_trials);
all_pos = zeros(n_pos, n_time, n_trials);

for i = 1:n_trials  % loop through all trials
    spikes_full = trial(i,angle).spikes(:,time_range);
    pos_full = trial(i,angle).handPos(:,time_range);
    if size(spikes_full, 2) >= length(time_range)  % Ensure trial has enough timepoints
        all_spikes(:, :, i) = spikes_full;
        all_pos(:, :, i) = pos_full;
    end
end

% Sliding window loop
for w = 1:n_wind
    % Define window range (within 300:600)
    t_start = (w-1) * args.winStp + 1;
    t_end = t_start + args.winSz - 1;

    % Compute spike count per window for each trial
    spike_counts = sum(all_spikes(:, t_start:t_end, :), 2); % Sum over time window
    spike_counts = reshape(spike_counts, size(all_spikes, 1), size(all_spikes, 3)); % [neurons x windows]

    % Compute position per window for each trial
    pos_counts = sum(all_pos(:, t_start:t_end, :), 2); % Sum over time window
    pos_counts = reshape(pos_counts, size(all_pos, 1), size(all_pos, 3)); % [pos x windows]

    % Compute spike rate (spikes/sec)
    spike_rates = spike_counts / args.winSz * 1000; % Convert to Hz

    % Compute mean and variance across trials
    meanSpikeRate(:, w) = mean(spike_rates, 2);
    varSpikeRate(:, w) = var(spike_rates, 0, 2);
    meanHandPos(:, w) = mean(pos_counts,2);
    varHandPos(:, w) = var(pos_counts,0,2);

    % Store time bin center (relative to the time range)
    time_bins(w) = time_range(round((t_start + t_end) / 2));
end


end
