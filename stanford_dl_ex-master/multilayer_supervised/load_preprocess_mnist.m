function [data_train, labels_train, data_test, labels_test] = load_preprocess_mnist()
%% TODO ensure this is consistent with common loaders
% assumes relative paths to the common directory
% assumes common directory on paty for access to load functions
% adds 1 to the labels to make them 1-indexed
% data_train.X:784*60000  labels_train.y:60000*1
%  data_test.X:784*10000   labels_test.y:10000*1

data_train = loadMNISTImages('E:\SummerCourse\UFLDL\common\train-images-idx3-ubyte');
labels_train = loadMNISTLabels(['E:\SummerCourse\UFLDL\common\train-labels-idx1-ubyte']);
labels_train  = labels_train + 1;

data_test = loadMNISTImages('E:\SummerCourse\UFLDL\common\t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels(['E:\SummerCourse\UFLDL\common\t10k-labels-idx1-ubyte']);
labels_test = labels_test + 1;

