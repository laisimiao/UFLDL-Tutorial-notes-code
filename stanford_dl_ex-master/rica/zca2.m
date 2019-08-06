function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size 81*10000
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
avg = mean(x, 1);     % Compute the mean pixel intensity value separately for each patch. 
x = x - repmat(avg, size(x, 1), 1);
sigma = x * x' / size(x, 2); % covariance matrix
[U,S,~] = svd(sigma);
xRot = U' * x;          % rotated version of the data.
Z = U * diag(1./sqrt(diag(S) + epsilon)) * xRot;