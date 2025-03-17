function [W, projMat] = applyLDA(reducedFeatures, X_pca, data)

    [n, k] = size(data);
    dataMean = zeros(size(X_pca,1),k);
    % Get between-class and within-class scatter matrices

    for angle = 1: k
        dataMean(:,angle) =  mean(X_pca(:,n*(angle-1)+1:angle*n),2);
    end

    Sb = (dataMean - mean(X_pca,2))*(dataMean - mean(X_pca,2))'; % Between class scatter matrix
    x_grand =  (X_pca - mean(X_pca,2))*(X_pca - mean(X_pca,2))'; % Grand mean
    Sw = x_grand - Sb; % Within class scatter matrix

    pcaDim = 200; % Keep the first 200 components from PCA
    Sw = reducedFeatures(:,1:pcaDim)' * Sw * reducedFeatures(:,1:pcaDim);
    Sb = reducedFeatures(:,1:pcaDim)' * Sb * reducedFeatures(:,1:pcaDim);

    % Ensure Sw is invertible
    if rank(Sw) < size(Sw, 1)
        Sw = pinv(Sw);  % Use pseudo-inverse if singular
    else
        Sw = inv(Sw);   % Regular inverse
    end

    [eigVecLDA, eigValLDA] = eig(Sw*Sb);

    [~,sortIdx] = sort(diag(eigValLDA),'descend');

    % optimum output
    projMat = reducedFeatures(:,1:pcaDim)*eigVecLDA(:,sortIdx(1:2));
    % optimum projection from the Most Discriminant Feature Method....!
    W = projMat' * (X_pca - mean(X_pca,2));
end
%% Plot
colors = {[1 0 0],[0 1 1],[1 1 0],[0 0 0],[0 0.75 0.75],[1 0 1],[0 1 0],[1 0.50 0.25]};
figure
hold on
for i=1:k
    plot(W(1,n*(i-1)+1:i*n),W(2,n*(i-1)+1:i*n),'o','Color',colors{i},'MarkerFaceColor',colors{i},'MarkerEdgeColor','k')
    hold on
end

legend('30','70','110','150','190','230','310','350');
%reachAngles = [30 70 110 150 190 230 310 350];
