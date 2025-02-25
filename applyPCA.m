function [pcaParams, reducedFeatures] = applyPCA(X)

% IMPLEMENTATION : [pcaParams, reducedFeatures] = applyPCA(spikeRates);
% INPUT: m by n matrix
% OUTPUT: reducedFeatures --> principal components
%         pcaParams --> projectionMatrix, explainedVar, numComponents (number of PCA components), mean, cumulative variance
          
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

    % Calculate cuumulative variance
    cumVar=cumsum(explainedVar);
    
    % Select number of components that explain 95% of variance and ensure
    % we have at least 2 components
    numComponents=find(cumVar>=0.95, 1);
    numComponents=max(numComponents,2)
    
    % Get projection matrix
    pcaParams.projectionMatrix=eigVectors(:, 1:numComponents);
    pcaParams.explainedVar=explainedVar;
    pcaParams.numComponents=numComponents;
    pcaParams.meanX=mean(X, 1);
    pcaParams.cumVar=cumVar;
    
    % Project data onto principal components
    reducedFeatures=X*pcaParams.projectionMatrix;
end
    
