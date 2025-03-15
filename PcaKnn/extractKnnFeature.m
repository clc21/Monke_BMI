function [X, y] = extractKnnFeature(smoothedSpikeRate, knnFeature)

[n,k] = size(smoothedSpikeRate);    

% Initialise
X = [];                % feature matrix
y = [];                % label vector

for i = 1:n  
    for j = 1:k
        spikeRate = smoothedSpikeRate{i, j};
        if strcmp(knnFeature, 'meanSpikeRate')
            % Use the actual mean per neuron as features
            knnFeatureVector = mean(spikeRate, 2)';  % Transpose to make a row vector
        else
            % Use variance per neuron as features
            knnFeatureVector = var(spikeRate, 0, 2)';
        end

        X = [X; knnFeatureVector];
        y = [y; j];
    end
end



end