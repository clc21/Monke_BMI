function [pcaParams, reducedFeatures] = applyPCA(X,N)

% IMPLEMENTATION : [pcaParams, reducedFeatures] = applyPCA(spikeRates);
% INPUT: m by n matrix - observations (n samples) and number of features (m variables)
%        
% OUTPUT: reducedFeatures --> principal components
%         pcaParams --> projectionMatrix, explainedVar, numComponents (number of PCA components), mean, cumulative variance
arguments
    X 
    N double = size(X,1)
end
    % Normalize data
    X=X-mean(X,1);
    
    % Compute covariance matrix
    covMatrix=cov(X);
    
    % Compute eigenvalues and eigenvectors
    [eigVectors,eigValues]=eig(covMatrix);
    eigValues=diag(eigValues);
    
    % Sort eigenvalues in descending order
    [eigValues, sortIdx]=sort(eigValues, 'descend');
    eigVectors=eigVectors(:, sortIdx);
    
    % Calculate explained variance ratio
    explainedVar=100*eigValues/sum(eigValues);

    % Calculate cumulative variance
    cumVar=cumsum(explainedVar);
    
    % Select number of components that explain 95% of variance and ensure
    % we have at least 2 components
    numComponents=find(cumVar>=95, 1, 'first');
    numComponents=max(numComponents,2);
    
    % Get projection matrix
    pcaParams.projectionMatrix=eigVectors(:, 1:numComponents);
    pcaParams.explainedVar=explainedVar;  % percentage of explained variance
    pcaParams.numComponents=numComponents;
    pcaParams.meanX=mean(X, 1);
    pcaParams.cumVar=cumVar;
    
    % Project data onto principal components
    reducedFeatures=X*pcaParams.projectionMatrix;
end
