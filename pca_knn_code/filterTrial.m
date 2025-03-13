function [filtered_data] = filter(data, matchLengthMethod)
    arguments
        data struct
        matchLengthMethod char {mustBeMember(matchLengthMethod, {'max', 'min'})} = 'max'
    end
    
    % Check if the data has a 'trial' field
    if ~isfield(data, 'trial')
        error('Input data does not have a "trial" field');
    end
    
    % Get dimensions of the trial struct array
    [n_trials, n_angles] = size(data.trial);
    
    % Initialize output with the same structure as input
    filtered_data = data;
    
    % Preallocate trial_lengths for better performance
    trial_lengths = zeros(n_trials * n_angles, 1);
    lengths_count = 0;
    
    % Determine all valid lengths
    for trial_idx = 1:n_trials
        for angle_idx = 1:n_angles
            total_time = size(data.trial(trial_idx, angle_idx).spikes, 2);
            start_idx = 301; 
            end_idx = total_time - 100;
            
            if end_idx > start_idx
                lengths_count = lengths_count + 1;
                trial_lengths(lengths_count) = end_idx - start_idx + 1;
            end
        end
    end
    
    % Trim any unused elements in the array
    trial_lengths = trial_lengths(1:lengths_count);
    
    % Get target length in one operation
    if strcmp(matchLengthMethod, 'max')
        target_length = max(trial_lengths);
    else
        target_length = min(trial_lengths);
    end
    
    % Process all trials
    for trial_idx = 1:n_trials
        for angle_idx = 1:n_angles
            current_trial = data.trial(trial_idx, angle_idx);
            total_time = size(current_trial.spikes, 2);
            start_idx = 301;
            end_idx = total_time - 100;
            
            if end_idx <= start_idx
                continue;
            end
            
            % Process spike data
            if isfield(current_trial, 'spikes')
                extracted_spikes = current_trial.spikes(:, start_idx:end_idx);
                [num_channels, current_length] = size(extracted_spikes);
                
                if current_length < target_length
                    % Pad with zeros - preallocate for speed
                    adjusted_spikes = zeros(num_channels, target_length);
                    adjusted_spikes(:, 1:current_length) = extracted_spikes;
                else
                    % Truncate to target length
                    adjusted_spikes = extracted_spikes(:, 1:target_length);
                end
                
                filtered_data(trial_idx, angle_idx).spikes = adjusted_spikes;
            end
            
            % Process handPos data
            if isfield(current_trial, 'handPos')
                extracted_handPos = current_trial.handPos(:, start_idx:end_idx);
                [num_channels, current_length] = size(extracted_handPos);
                
                if current_length < target_length
                    % Pad with zeros - preallocate for speed
                    adjusted_handPos = zeros(num_channels, target_length);
                    adjusted_handPos(:, 1:current_length) = extracted_handPos;
                else
                    % Truncate to target length
                    adjusted_handPos = extracted_handPos(:, 1:target_length);
                end
                
                filtered_data(trial_idx, angle_idx).handPos = adjusted_handPos;
            end
        end
    end
end
