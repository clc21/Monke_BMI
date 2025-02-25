classdef kNNeighbors

properties
    k = 5
    % euclidean, mahalanobis, minkowski, chebychev, cosine, correlation, hamming, jaccard 
    X
    name_labels
    Y
end
methods (Access = public)
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
        distance
        % code here
    end
    function Ypred = predict(obj,Xnew)
        % code here
    end
end
end