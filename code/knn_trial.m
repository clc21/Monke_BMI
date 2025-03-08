clc; close all; clear all;
%% KNN
% 1. Select an optimal k value
% 2. Calc distance
% 3. Find nearest neighbour
% 4. Return most common class
%% Selecting k - Cross-Validation 
% 1. split data into k_folds subsets
% 2. for each fold, train the model on k_folds - 1 subsets and validate
% measure the accuracy for different values of k
% 3. select k that has the best average accuracy across folds.

%% Obtain Y (Label)

function labels = generateLabels(trial_data)
    [n,k] = size(trial);
    labels = [];
    for angle_idx = 1:k
        for trials_idx = 1:n
            labels = [labels; angle_idx]; % Assign reaching angle as labels
        end
    end     
end
%% K-fold Cross Validation
% 'Y' is the output vector or reponse (N-by-1) = labels
% 'X' is the matrix of input vectors ((N-by-D)= training data
% 'X' is the input data from preprocessing

function CVerror = crossValidation(X, labels)
    % Function to perform K-fold cross-validation and find the optimal k for RBF network
    % Input:
    %   X - Matrix of input vectors (N-by-D) from training data
    %   labels - Output vector (N-by-1) containing the labels (targets)
    % Output:
    %   best_k - Optimal k value
 
    % Data
    N = size(X, 1);  % Number of trials
    D = size(X, 2);  % Number of features - angles
    x = X';          % transpose X to match input format
    y = labels';     % tranpose Y to match input format
    goal = 0.000001; % Sum-squared error goal
 
    % Params
    k_values = [1, 3, 5, 7, 9, 11, 13, 15];       % Different odd k values to test
    num_k = length(k_values);                     % Number of k values
    sc = [0.1, 0.5, 1, 2, 5, 10];                 % spread constant for radial basis function network
    M = size(sc, 2);                              % Number of spread constants

    CVerror = zeros(num_k, M);                    % Store cross-validation errors
    Indices = crossvalind('Kfold', N, 10);        % Split data into 10 folds           for fold = 1:10

    for j = 1:M
       for i = 1:num_k
           sse = 0;                                          % Initialize squared-error
           for fold = 1:10                                   % 10-Fold Cross-Validation
              test = (Indices == fold); train = ~test;
              net = newrb(x(1:D,train),y(train),goal,sc(j)); % Train rbf network
              yhat = sim(net,x(:,test));                     % Obtain predictions for test set
              sse = sse + sum((yhat' - Y(test)).^2);         % Calc squared-error
           end
           CVerror (i,j) = sse / 10;                         % Mean-squared error
       end
    end
end
    
 function best_k = findBestK(CVerror)
    [~, best_k_idx] = min(CVerror(:));  % Find the best k (minimum cross-validation error)
    [best_k_row, ~] = ind2sub(size(CVerror), best_k_idx);
    best_k = k_values(best_k_row);
end
%% Cross-Validation With In-Built Functions

function best_k = kFoldCrossValidation(X, labels)
    % 'Y' is the output vector or reponse (N-by-1) = labels
    % 'X' is the matrix of input vectors ((N-by-D)= training data
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]; 
    cv_errors = zeros(length(k_values), 1);

    % Perform 10-fold cross-validation
    for i = 1:length(k_values)
        Mdl = fitcknn(X, Y, 'NumNeighbors', k_values(i), 'CrossVal', 'on', 'KFold', 10);
        cv_errors(i) = kfoldLoss(Mdl);  % Compute classification error
    end

    % Find the best k with minimum error
    [~, best_k_idx] = min(cv_errors);
    best_k = k_values(best_k_idx);
end
%% KNN
% properties
%     k = 5
%     metric = 'euclidean'
%     % euclidean, mahalanobis, minkowski, chebychev, cosine, correlation, hamming, jaccard 
%     X           % training data
%     name_labels
%     Y           % label
% end
function obj = kNNeighbors(k,metric)
    if nargin > 0
        obj.k = k;
    end
    if nargin > 1
        obj.metric = metric;
    end
end
function obj = fit(obj,X,Y)
    obj.X = X;
    [obj.name_labels,~,obj.Y] = unique(Y); % Numeric labels
end
function [distances,indices] = find(obj,Xnew)
    distance = pdist2(obj.X, Xnew, obj.metric);
    [distances, indices] = sort(distance);
    distances = distances(1: obj,k, :);
    indices = indices(1:obj.k, :);
end
function Ypred = predict(obj,Xnew)
    [~, indices] = find(obj, Xnew);
    Ynearest = obj.Y(indices); % k-nearest labels
    dim = 2-(obj.k > 1);       % dimension for the mode function
    Ypred = obj.name_labels(mode(Ynearest, dim)); % most frequent label
end



