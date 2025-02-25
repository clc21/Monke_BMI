function [pcaParams, reducedFeatures] = applyPCA(X)
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
    
    % Select number of components that explain 95% of variance
    numComponents=find(explainedVar>=0.95, 1);

    % Calculate cuumulative variance
    cumVar=cumsum(explainedVar);
    
    % Get projection matrix
    pcaParams.projectionMatrix=eigVectors(:, 1:numComponents);
    pcaParams.explainedVar=explainedVar;
    pcaParams.numComponents=numComponents;
    pcaParams.meanX=mean(X, 1);
    pcaParams.cumVar=cumVar;
    
    % Project data onto principal components
    reducedFeatures=X*pcaParams.projectionMatrix;
end
