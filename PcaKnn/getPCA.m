function [pcaParams_mat, reducedFeatures_mat] = getPCA(data)
    arguments
        data struct
    end
    
    [n,k] = size(data);                % n = num_trials, k = angles
    pcaParams_mat = cell(n, k);        % Store PCA parameters for each angle
    reducedFeatures_mat = cell(n, k);  % Store PCA-reduced features

    for angle = 1:k
        for t = 1: n
            %X = data(t,angle);
            %N = size(X,1);
            
            % extract spike rate
            [spikeRate,~] = extractWindows(data, t,angle);
            
            % apply gaussian smoothing
            spikeRate = applyGaussianFilter(spikeRate);
            
            % Apply PCA
            [pcaParams, reducedFeatures] = applyPCA(spikeRate);

            % Store the results
            pcaParams_mat{t, angle} = pcaParams;
            reducedFeatures_mat{t, angle} = reducedFeatures;
        end
    end
end