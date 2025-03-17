function [X, y, Xnew, Ynew, labels] = extractKnnFeature(W, data)

    angle = size(data, 2);
    trialDir = size(W, 2); % 800 samples
   
    % Initialise
    X = [];                % feature matrix
    y = [];                % label vector
    Xnew = [];
    Ynew = [];
       
    %X_transformed = X_afterPCA(:,1:pcaDim)' * (X_pca - mean(X_pca,2));
    labels = repelem(1:angle, trialDir/angle)';

    % Split data into 80% training and 20% testing
    ratioTrain = 0.8; % 80% training, 20% testing
    numTrain = round(ratioTrain * trialDir);

    randIdx = randperm(trialDir); % Shuffle indices
    trainIdx = randIdx(1:numTrain);
    testIdx = randIdx(numTrain+1:end);

    X = W(:, trainIdx)';
    Xnew = W(:, testIdx)';
    y = labels(trainIdx);
    Ynew = labels(testIdx);

end