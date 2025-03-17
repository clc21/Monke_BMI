function [pcaParams_mat, reducedFeatures_mat, spikeRateMat] = getSmoothedSpikeRate(data)
    arguments
        data struct
    end
    
    [n,k] = size(data);                % n = num_trials, k = angles
    pcaParams_mat = cell(n, k);        % Store PCA parameters for each angle
    reducedFeatures_mat = cell(n, k);  % Store PCA-reduced features
    spikeRateMat = cell(n, k);

    for angle = 1:k
        for t = 1:n
            % Call extractWindows with all required arguments
            [spikeRate, ~] = extractWindows(data, t, angle);

            % Apply gaussian smoothing
            spikeRate = applyGaussianFilter(spikeRate);

            % Store spikeRate into a struct
            spikeRateMat{t, angle} = spikeRate;

            % Apply PCA
            %[pcaParams, reducedFeatures] = obj.applyPCA(spikeRate);

            % Store the results
            %pcaParams_mat{t, angle} = pcaParams;
            %reducedFeatures_mat{t, angle} = reducedFeatures;
 
        end
    end
end