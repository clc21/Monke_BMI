function [Ypred, accuracy, best_k] = knn_crossVal(X, y, Xnew, Ynew)
    arguments
        X    % Feature vector of training data
        y    % Labels (angle)
        Xnew % Feature vector of test data
        Ynew % Labels of test data
    end

    % Select the optimal k (best_k)
    k_val = 1:10;
    cv_errors = zeros(size(k_val));

    for i = 1:length(k_val)
        knnModel = fitcknn(X, y, 'NumNeighbors', k_val(i), 'CrossVal', 'on', 'KFold', 10);
        % y_pred = predict(model, X_test);
        cv_errors(i) = kfoldLoss(knnModel); % compute cross-validation error
    end

    % Find the best k (minimum error)
    [~, best_k_idx] = min(cv_errors);
    best_k = k_val(best_k_idx);
    
    if mod(best_k, 2) == 0
        best_k = best_k + 1;  % If k is even, add 1 to make it odd
    end

    % Train final knn model with best_k
    best_knnModel = fitcknn(X, y, 'NumNeighbors', best_k);
    Ypred = predict(best_knnModel, Xnew);

    % Accuracy
    accuracy = sum(Ypred == Ynew) / length(Ynew) * 100;

end