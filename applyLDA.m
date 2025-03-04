function [ldaModel] = applyLDA(X, y)
  % Implementation of Linear Discriminant Analysis
  % X: feature matrix (samples x features)
  % y: class labels
  
  % Get unique classes
  classes = unique(y);
  numClasses = length(classes);
  [numSamples, numFeatures] = size(X);
  
  % Compute class means
  classMeans = zeros(numClasses, numFeatures);
  for i = 1:numClasses
      classMeans(i, :) = mean(X(y == classes(i), :), 1);
  end
  
  % Compute global mean
  globalMean = mean(X, 1);
  
  % Compute between-class scatter matrix
  Sb = zeros(numFeatures, numFeatures);
  for i = 1:numClasses
      n_i = sum(y == classes(i));
      meanDiff = classMeans(i, :) - globalMean;
      Sb = Sb + n_i * (meanDiff' * meanDiff);
  end
  
  % Compute within-class scatter matrix
  Sw = zeros(numFeatures, numFeatures);
  for i = 1:numClasses
      classIdx = (y == classes(i));
      classData = X(classIdx, :);
      classMeanMatrix = repmat(classMeans(i, :), sum(classIdx), 1);
      Sw = Sw + (classData - classMeanMatrix)' * (classData - classMeanMatrix);
  end
  
  % Compute the transformation matrix by solving Sb*w = lambda*Sw*w
  % This is equivalent to finding eigenvectors of inv(Sw)*Sb
  % Note: Using pinv for potential numerical stability with high-dim data
  [V, D] = eig(pinv(Sw) * Sb);
  
  % Sort eigenvectors by eigenvalues in descending order
  [~, sortIdx] = sort(diag(D), 'descend');
  V = V(:, sortIdx);
  
  % Keep only the eigenvectors for n_classes - 1 dimensions
  V = V(:, 1:min(numClasses-1, numFeatures));
  
  % Project class means to LDA space
  projectedMeans = classMeans * V;
  
  % Return the LDA model
  ldaModel.V = V;                      % Projection matrix
  ldaModel.classMeans = classMeans;    % Class means in original space
  ldaModel.projectedMeans = projectedMeans; % Class means in LDA space
  ldaModel.classes = classes;          % Class labels
  ldaModel.globalMean = globalMean;    % Global mean

  % Project the data onto LDA space
  % X_lda = (X - ldaModel.globalMean) * ldaModel.V;
end