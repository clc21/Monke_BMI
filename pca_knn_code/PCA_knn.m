clc; clear all;

before = load('monkeydata_training.mat');

%afterMax = filterTrial(before,'max');
afterMin = filterTrial(before,'min');

%afterPCA_max = getPCA(afterMax);
afterPCA_min = getPCA(afterMin);

[X_train, y_train] = extractKnnFeature(afterPCA_min, 'meanSpikeRate');

[Ypred, k, accuracy] = applyKNN(X_train, y_train, X_train, afterPCA_min);