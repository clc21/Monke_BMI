
%% Main
clc; close all; clear all;

data = load('monkeydata_training.mat');
% Filter out the 300ms before and 100ms after trial data 
filtered_data = filterTrial(data, 'min');
% Extract spike rate and apply Gaussian Smoothing to it
[~, ~, spikeRateMat] = getSmoothedSpikeRate(filtered_data);
% Reformat the spike rate matrix to suit PCA 
%(numNeurons*timebins x numTrials*angle)
X_pca = inputPCA(spikeRateMat);

% Apply PCA to the reformated spike rate matrix
[~, afterPCA, covMatrix] = applyPCA(X_pca);
% Perform LDA onto the principle components to get the mean position of
% the neurons
[W, projMat] = lda(afterPCA, X_pca, filtered_data);
% Classify / group the neurons using KNN (give an angle value)
[X, y, Xnew, Ynew, labels] = extractKnnFeature(W, filtered_data); % extract X and y for knn input
[Ypred, accuracy, best_k] = knn_crossVal(X, y, Xnew, Ynew);
