 function [eigValues,reducedFeatures, covMatrix] = applyPCA(X)

% IMPLEMENTATION : [pcaParams, reducedFeatures] = applyPCA(spikeRates);
% INPUT: m by n matrix - observations (n samples) and number of features (m variables)
%        
% OUTPUT: reducedFeatures --> principal components

    % Normalize data by subtract cross-trial mean
    X = X - mean(X, 2);

    % Compute covariance matrix
    covMatrix = (X' * X)/(size(X, 2));

    % Compute eigenvalues and eigenvectors
    [eigVectors, eigValues] = eig(covMatrix);

    % Sort eigenvalues in descending order
    [~, sortIdx] = sort(diag(eigValues), 'descend');
    eigVectors = eigVectors(:, sortIdx);
    
    % Get eigenvalues and not the zeros of the diagonal matrix
    eigValues = diag(eigValues);
    eigValues = diag(eigValues(sortIdx));

    % project data onto the newly derived basis and normalise it
    pc = X*eigVectors;
    reducedFeatures = pc./sqrt(sum(pc.^2));

end
    
