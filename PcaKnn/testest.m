%% Example usage of KnnAngleDecoder class
% This script demonstrates how to use the streamlined KnnAngleDecoder class

% Clear workspace
clc; clear all;

% Load the data
data = load('monkeydata_training.mat');

% Create a decoder with default parameters
decoder = KnnMethod();

% Run the complete pipeline and get only the needed outputs
[Ypred, accuracy, best_k] = decoder.runPipeline(data, 'min');

fprintf('Angle Classification Results:\n');
fprintf('Best k: %d\n', best_k);
fprintf('Accuracy: %.2f%%\n', accuracy);

%% 