classdef knnMethods
    properties
        k      % Number of neighbours
        metric % Distance metric
        X      % Training data
        name_labels
        Y      % Labels  
    end
    
    methods
        
        function obj = knnMethods(k,metric)
    % USAGE:
    %    metric = 'euclidean', 'cityblock', 'minkowski', 'cosine',
    %    'correlation', 'hamming', etc.

            obj.k = k;
            obj.metric = metric;
        end

        function obj = fit(obj, X, Y)  % fit the model
            obj.X = X;
            [obj.name_labels, ~, obj.Y] = unique(Y);     % numeric labels
        end

        function [distances,indices] = findKNN(obj,Xnew) % find k-nearest neighbours
            distances = pdist2(obj.X,Xnew,obj.metric);   % Euclidean distance matrix
            [distances,indices] = sort(distances);       % Ordered distances
            distances = distances(1:obj.k,:);
            indices = indices(1:obj.k,:);
        end

        function Ypred = predict(obj,Xnew)            % predict labels
            [~,indices] = findKNN(obj,Xnew);
            Ynearest = obj.Y(indices);                % k-nearest labels
            dim = 2 - (obj.k > 1);                    % dimension for the mode function
            Ypred = obj.name_labels(mode(Ynearest,dim)); % Most frequent label
        end
    end

end
