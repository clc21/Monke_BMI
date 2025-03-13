function [X, y] = extractKnnFeature(data, knnFeature)
% INPUT: 
%   trial   = struct - observations containing neuron spike data
% OUTPUT: 
%   featureMatrix - of mean spike count

arguments
    data 
    knnFeature char {mustBeMember(knnFeature, {'meanSpikeRate', 'varSpikeRate'})} = 'meanSpikeRate'
end


[n,k] = size(data);    % number of trials for one reach angle

% Initialise
X = [];                % feature matrix
y = [];                % label vector

for i = 1:n  
    for j = 1:k
        spike_data = data{i,j}.projectionMatrix;
        
        if strcmp(knnFeature, 'meanSpikeRate')
            knnFeatureVector = mean(spike_data, 2);
        else
            knnFeatureVector = var(spike_data, 2);
        end
    
        X = [X, knnFeatureVector];
        y = [y, j];
    end
end



end