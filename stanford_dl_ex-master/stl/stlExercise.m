%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. ����û�ж�Ӧ�ģ���Ӧ�����ϰ��
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your RICA to get good filters; you do not need to 
%  change the parameters below.
clear;close all;
addpath(genpath('E:\SummerCourse\UFLDL\stanford_dl_ex-master\common')) % path to minfunc
imgSize = 28;
global params;
params.patchWidth=9;           % width of a patch
params.n=params.patchWidth^2;   % dimensionality of input to RICA
params.lambda = 0.0005;   % sparsity cost
params.numFeatures = 32; % number of filter banks to learn
params.epsilon = 1e-2;   

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
mnistData   = loadMNISTImages('E:\SummerCourse\UFLDL\common\train-images-idx3-ubyte'); % 784*60000
mnistLabels = loadMNISTLabels('E:\SummerCourse\UFLDL\common\train-labels-idx1-ubyte'); % 60000*1

numExamples = size(mnistData, 2);
% 50000 of the data are pretended to be unlabelled
unlabeledSet = 1:50000;
unlabeledData = mnistData(:, unlabeledSet);

% the rest are equally splitted into labelled train and test data


trainSet = 50001:55000;
testSet = 55001:60000;
trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-10
% only keep digits 0-4, so that unlabelled dataset has different distribution
% than the labelled one.
removeSet = find(trainLabels > 5);
trainData(:,removeSet)= [] ;
trainLabels(removeSet) = [];

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-10
% only keep digits 0-4
removeSet = find(testLabels > 5);
testData(:,removeSet)= [] ;
testLabels(removeSet) = [];


% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set trainData: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set testData: %d\n\n', size(testData, 2));

%% ======================================================================
%  STEP 2: Train the RICA
%  This trains the RICA on the unlabeled training images. 

%  Randomly initialize the parameters
randTheta = randn(params.numFeatures,params.n)*0.01;  % 1/sqrt(params.n); 32*81
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2)); 
randTheta = randTheta(:); % 2591

% subsample random patches from the unlabelled+training data,�����°�̳���˵ֻ��unlabelled��
% patches = samplePatches([unlabeledData,trainData],params.patchWidth,200000); % 81*200000
patches = samplePatches(unlabeledData,params.patchWidth,200000); % 81*200000

%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 1000;
% You'll need to replace this line with RICA training code
% opttheta = randTheta;

%  Find opttheta by running the RICA on all the training patches.
%  You will need to whitened the patches with the zca2 function 
%  then call minFunc with the softICACost function as seen in the RICA exercise.
%%% YOUR CODE HERE %%%
patches = zca2(patches); 
m = sqrt(sum(patches.^2) + (1e-8));
x = bsxfunwrap(@rdivide,patches,m);
% ����֮ǰ��ΪsoftICACost.m���������lambdaд����1��̫���ˣ�����û�������ξ�ͣ������
% ��Сһ���Ժ�Ϳ����ˣ���ȷ��Ҳ�����ˣ��ﵽ�˽̳��еı�׼����W'��ͼҲ�ܿ����������
tic;
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost(theta, x, params), randTheta, options); 
fprintf('# Optimization took: %f seconds.\n', toc);
% reshape visualize weights
W = reshape(opttheta, params.numFeatures, params.n); % 32*81
display_network(W');

%% ======================================================================

%% STEP 3: Extract Features from the Supervised Dataset
% pre-multiply the weights with whitening matrix, equivalent to whitening
% each image patch before applying convolution. V should be the same V
% returned by the zca2 when you whiten the patches.
% W = W*V; % V��ɶ��һ���±ƣ���ע�͵���˵
%  reshape RICA weights to be convolutional weights.
W = reshape(W, params.numFeatures, params.patchWidth, params.patchWidth);
W = permute(W, [2,3,1]); % patchWidth * patchWidth * numFeatures

%  setting up convolutional feed-forward. You do need to modify this code.
filterDim = params.patchWidth;
poolDim = 5;
numFilters = params.numFeatures;
trainImages=reshape(trainData, imgSize, imgSize, size(trainData, 2));
testImages=reshape(testData, imgSize, imgSize, size(testData, 2));
%  Compute convolutional responses
%  TODO: You will need to complete feedfowardRICA.m ������������� ����������ػ������������
trainAct = feedfowardRICA(filterDim, poolDim, numFilters, trainImages, W);
fprintf('# ��2500�ص�500����Ϊ�����ˣ��������һ��feedfowardRICA\n');
testAct = feedfowardRICA(filterDim, poolDim, numFilters, testImages, W);
%  reshape the responses into feature vectors
featureSize = size(trainAct,1)*size(trainAct,2)*size(trainAct,3); % 512
trainFeatures = reshape(trainAct, featureSize, size(trainData, 2)); % 512*2538
testFeatures = reshape(testAct, featureSize, size(testData, 2)); %512*2520
%% ======================================================================
%% STEP 4: Train the softmax classifier

numClasses  = 5; % doing 5-class digit recognition
% initialize softmax weights randomly
randTheta2 = randn(numClasses, featureSize)*0.01;  % 1/sqrt(params.n);
randTheta2 = randTheta2 ./ repmat(sqrt(sum(randTheta2.^2,2)), 1, size(randTheta2,2)); 
randTheta2 = randTheta2';
randTheta2 = randTheta2(:);

%  Use minFunc and softmax_regression_vec from the previous exercise to 
%  train a multi-class classifier. 
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 300;

% optimize
%%% YOUR CODE HERE %%%
[opt_theta, ~, ~] = minFunc(@softmax_regression_vec, randTheta2, options, trainFeatures, trainLabels);
opt_theta = reshape(opt_theta,featureSize,numClasses);
opt_theta = opt_theta'; % numClasses * featureSize
%%======================================================================
%% STEP 5: Testing 
% Compute Predictions on tran and test sets using softmaxPredict
% and softmaxModel�����а��������ϰ�ģ�
%%% YOUR CODE HERE %%%
[~,train_pred] = max(opt_theta * trainFeatures); % opt_theta��2560*1(��randTheta2��sizeһ��) trainFeatures:512*2538
[~,pred] = max(opt_theta * testFeatures); % testFeatures:512*2520

% Classification Score
fprintf('Train Accuracy: %f%%\n', 100*mean(train_pred(:) == trainLabels(:))); % trainLabels:1*2538
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:))); % testLabels:1*2520
% You should get 100% train accuracy and ~99% test accuracy. With random
% convolutional weights we get 97.5% test accuracy. Actual results may
% vary as a result of random initializations

