% Load training data
load('monkeydata_training.mat');

% Number of angles and neurons
angles = 1:8;
neurons = 1:98;
num_trials = 100;

% Initialize firing rates (lambda) for each neuron and angle
lambda = zeros(98, 8);

% Calculate firing rates using training data
for a = angles
    for n = neurons
        spike_counts = zeros(1, num_trials);
        for trial_num = 1:num_trials
            spike_counts(trial_num) = sum(trial(trial_num, a).spikes(n, :));
        end
        lambda(n, a) = mean(spike_counts); % average spike count
    end
end

% Ensure no zeros in lambda to avoid log(0)
lambda = lambda + eps; % add small constant

% Bayesian classification
predicted_angles = zeros(num_trials, 8); % Store predictions for each trial

for a = angles
    for trial_num = 1:num_trials
        new_spikes = trial(trial_num, a).spikes; % spike data for current trial
        log_posteriors = zeros(1, 8);
        
        % Calculate log posterior for each angle
        for a_test = angles
            log_likelihood = 0;
            for n = neurons
                k = sum(new_spikes(n, :)); % spike count for neuron n
                % Poisson log likelihood: k * log(lambda) - lambda
                log_likelihood = log_likelihood + (k * log(lambda(n, a_test)) - lambda(n, a_test));
            end
            % Uniform prior P(angle) = 1/8 -> log(1/8)
            log_posteriors(a_test) = log_likelihood + log(1/8);
        end
        
        % Predicted angle = argmax of log posterior
        [~, predicted_angle] = max(log_posteriors);
        predicted_angles(trial_num, a) = predicted_angle;
    end
end

% Display predictions
disp('Predicted angles for each trial:')
disp(predicted_angles)

% Convert matrix to vector
Ypred = predicted_angles(:);  % Flatten predicted angles into a vector
Ynew = repmat(angles, num_trials, 1); % Create ground truth labels
Ynew = Ynew(:); % Flatten into vector

% Compute accuracy
accuracy = mean(Ypred == Ynew); % Calculate accuracy as (correct predictions / total predictions)

% Display results
fprintf('Classification Accuracy: %.2f%%\n', accuracy * 100);
