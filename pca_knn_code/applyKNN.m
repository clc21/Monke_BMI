function [Ypred, k, accuracy] = applyKNN(X, y, Xnew, data)
% INPUT: 
%   X = feature vector of training data
%   y = labels vector
%   Xnew = feature vector of test data

% OUTPUT: 
%   y_pred  = predicted reaching angles
%   best_k  = optimal k value for k-NN
%   accuracy = classification accuracy
    
    k = round(sqrt(size(data, 1)));
    % Ensure k is odd
    if mod(k, 2) == 0
        k = k + 1;  % If k is even, add 1 to make it odd
    end
    if k > 5
        k = 5
    end
    
    knn = kNNeighbours(k, 'euclidean');
    knn = knn.fit(X, y) % fit the knn model
    [distances, indices] = knn.find(Xnew); % find k-nearest neighbours
    Ypred = knn.predict(Xnew); % predict labels (reaching angle)
    
    % Accuracy
    Ypred = Ypred(:); % turn Ypred into a column vector
    accuracy = sum(Ypred == y) / length(y) * 100;
    
end