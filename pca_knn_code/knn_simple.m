arguments
    trial struct
end

% Number of angles and neurons
[n,k] = size(trial); % n = num_trials
angles = 1:k;
num_neurons = 1:size(trial(1,1).spikes, 1); % 98 neurons

% ~k-fold cross validation (10-fold)~ %
X_train = []; % feature vector
y_train = []; % label (ie. angles)


for angle = angles
    for t = 1: n  
        [meanSpikeRate,~] = extract(trial,angle);
        meanSpikeRate = mean(meanSpikeRate, 2)'; % average to get a single feature vector
        X_train = [X_train; meanSpikeRate];
        y_train = [y_train; angle];    
    end
end

% Select the optimal k (best_k)
k_val = 1:10;
cv_errors = zeros(size(k_val));

for i = 1:length(k_val)
    knnModel = fitcknn(X_train, y_train, 'NumNeighbors', k_val(i), 'CrossVal', 'on', 'KFold', 10);
    % y_pred = predict(model, X_test);
    cv_errors(i) = kfoldLoss(knnModel); % compute cross-validation error
end

% Find the best k (minimum error)
[~, best_k_idx] = min(cv_errors);
best_k = k_val(best_k_idx);

% Train final knn model with best_k
best_knnModel = fitcknn(X_train, y_train, 'NumNeighbors', best_k);
y_pred = predict(best_knnModel, X_train);

% Accuracy
accuracy = sum(y_pred == y_train) / length(y_train) * 100;

end