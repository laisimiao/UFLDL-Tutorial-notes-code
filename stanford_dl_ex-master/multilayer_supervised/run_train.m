clear; close all; clc
% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath E:\SummerCourse\UFLDL\stanford_dl_ex-master\common;
addpath(genpath('E:\SummerCourse\UFLDL\stanford_dl_ex-master\common\minFunc_2012\minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();
% data_train.X:784*60000  labels_train.y:60000*1
%  data_test.X:784*10000   labels_test.y:10000*1

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,... % minFunc(funObj,x0,options,varargin)
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true); % [ cost, grad, pred_prob] = 
[~,pred] = max(pred);                                                     % supervised_dnn_cost( theta, ei, data, labels, pred_only)
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
