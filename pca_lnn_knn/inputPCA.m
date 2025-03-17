function [X_pca] = inputPCA(spikeRateData)

    arguments
        spikeRateData cell
    end

    [n, k] = size(spikeRateData);                       % n = num_trials, k = angles
    [numFeatures, timeBins] = size(spikeRateData{1,1}); % Get number of features (number of neurons)
    X_pca = zeros(numFeatures*timeBins, n*k);           % Preallocate matrix (800x784)

    
    for angle = 1:k
        for t = 1:n
            % Extract spike rate matri (nunNeurons x timebins)
            spikeRate = spikeRateData{t, angle};
            % Flatten the matric into a column vector
            flattenedVector = spikeRate(:);
            % Compute the column index in X_pca
            colIdx = (angle-1)*n + t;
            % Assign flattened vector to the corresponding column
            X_pca(:, colIdx) = flattenedVector;
            
        end
    end
    
end
