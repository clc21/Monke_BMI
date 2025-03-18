function [spikeRate,handKinematics,time_bins] = extractFeatures(trial,args)
%extractWindows Extract the spike rate, hand Pos&Vel&Acc in time windows
%   trial
arguments
    trial struct
    args.trialNumber double = 0
    args.angle double = 0
    args.isStruct logical = false
    args.winSz double = 10  % Window size in ms
    args.winStp double = 10 % Window step in ms
    args.sigma double = 50  % Standard deviation for Gaussian smoothing (in ms)
end

if args.isStruct
    spike_train = trial(args.trialNumber,args.angle).spikes;
    n_time = size(spike_train,2);
    try
        pos_array = trial(args.trialNumber,args.angle).handPos;
    catch
        pos_array = zeros(3,n_time);
    end
else
    spike_train = trial.spikes;
    n_time = size(spike_train,2);
    try
        pos_array = trial.handPos;
    catch
        pos_array = zeros(3,n_time);
    end
end

dt = 1e-3;                            % sampling period
n_neurons = size(spike_train,1);      % number of neurons
n_pos = size(pos_array,1);            % number of directions
time_range = 1:size(spike_train,2);  % slice of interest
n_wind = floor((n_time - args.winSz) / args.winStp) + 1;

% velocity is derivative of displacement
pos_prev = zeros(n_pos,n_time);
pos_prev(:,1) = pos_array(:,1);
pos_prev(:,2:end) = pos_array(:,1:end-1);
vel_array = (pos_array-pos_prev)/dt;

% acceleration is derivative of velocity\
vel_prev = zeros(n_pos,n_time);
vel_prev(:,1) = vel_array(:,1);
vel_prev(:,2:end) = vel_array(:,1:end-1);
acc_array = (vel_array-vel_prev)/dt;

% initialise arrays
spikeRate = zeros(n_neurons, n_wind);
handPos = zeros(n_pos,n_wind);      % average displacement per window
handVel = zeros(n_pos,n_wind);      % average velocity per window
handAcc = zeros(n_pos,n_wind);      % average acceleration per window
time_bins = zeros(1, n_wind);

% Sliding window loop
for w = 1:n_wind
    % Define window range (within 300:600)
    t_start = (w-1) * args.winStp + 1;
    t_end = t_start + args.winSz - 1;

    % Compute spike rate and kinematics
    spikeRate(:, w) = sum(spike_train(:, t_start:t_end), 2)/ args.winSz * 1000; % Convert to Hz
    handPos(:, w) = mean(pos_array(:, t_start:t_end),2);
    handVel(:, w) = mean(vel_array(:, t_start:t_end),2);
    handAcc(:, w) = mean(acc_array(:, t_start:t_end),2);

    % Store time bin center (relative to the time range)
    time_bins(w) = time_range(round((t_start + t_end) / 2));
end

% Apply Gaussian smoothing to spike rates
spikeRate = applyGaussianFilter(spikeRate, args.sigma, args.winSz);

handKinematics = cat(1,handPos(1:2,:),handVel(1:2,:),handAcc(1:2,:));

end

%% Function to Apply Gaussian Smoothing
function smoothedSpikeRate = applyGaussianFilter(spikeRate, sigma, winSz)
    % Create Gaussian kernel
    kernelSize = ceil(3 * sigma / winSz) * 2 + 1;  % Ensure it's odd
    t = linspace(-kernelSize / 2, kernelSize / 2, kernelSize);
    gaussKernel = exp(-t.^2 / (2 * sigma^2));
    gaussKernel = gaussKernel / sum(gaussKernel);  % Normalize

    % Apply convolution along the time dimension (2nd axis)
    smoothedSpikeRate = conv2(spikeRate, gaussKernel, 'same');
end
