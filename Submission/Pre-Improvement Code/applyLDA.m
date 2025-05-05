function [ldaModel] = applyLDA(X, y)
  % Implementation of Linear Discriminant Analysis
  % X: feature matrix (samples x features)
  % y: class labels
  
  % Get unique classes
  classes=unique(y);
  numClasses=length(classes);
  [numSamples, numFeatures]=size(X);
  
  % Compute class means
  classMeans = zeros(numClasses, numFeatures);
  classSizes = zeros(numClasses, 1);
  
    for i = 1:numClasses
        classIdx = (y == classes(i));
        classMeans(i, :) = mean(X(classIdx, :), 1);
        classSizes(i) = sum(classIdx);
    end
  
  % Compute global mean
  globalMean = mean(X, 1);
  
  % Compute between-class scatter matrix
  Sb = zeros(numFeatures, numFeatures);
  for i = 1:numClasses
      meanDiff = classMeans(i, :) - globalMean;
      Sb = Sb + classSizes(i) * (meanDiff' * meanDiff);
  end

  
  % Compute within-class scatter matrix
  Sw = zeros(numFeatures, numFeatures);
  for i = 1:numClasses
      classIdx = (y == classes(i));
      classData = X(classIdx, :);
      classMeanMatrix = repmat(classMeans(i, :), classSizes(i), 1);
      deviation = classData - classMeanMatrix;
      Sw = Sw + deviation' * deviation;
  end
  
  % Apply regularization to within-class scatter matrix
  lambda = 0.001; % Regularization parameter
  Sw_reg = Sw + lambda * trace(Sw) * eye(numFeatures);
  
  % Compute the transformation matrix by solving the generalized eigenvalue problem
  [V, D] = eig(Sb, Sw_reg);
  
  % Sort eigenvectors by eigenvalues in descending order
  [eigValues, sortIdx] = sort(diag(D), 'descend');
  V = V(:, sortIdx);
  
  % Keep only the eigenvectors for n_classes - 1 dimensions
  numComponents = min(numClasses-1, numFeatures);
  V = V(:, 1:numComponents);
  
  % Normalize the eigenvectors to make projections more comparable to PCA
  % This ensures the scale of projections is reasonable
  for i = 1:size(V, 2)
      V(:, i) = V(:, i) / norm(V(:, i));
  end
  
  % Project class means to LDA space
  projectedMeans = classMeans * V;
  
  % Return the LDA model
  ldaModel.V = V;                      % Projection matrix
  ldaModel.classMeans = classMeans;    % Class means in original space
  ldaModel.projectedMeans = projectedMeans; % Class means in LDA space
  ldaModel.classes = classes;          % Class labels
  ldaModel.globalMean = globalMean;    % Global mean
  ldaModel.eigenvalues = eigValues(1:numComponents); % Eigenvalues for components
end
