%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n); % 50*81

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
% cost Wgrad
lambda = 0.1;
epsilon = 1e-2;
cost = lambda * sum(sum(sqrt((W*x).^2+epsilon))) + sum(sum((W'*W*x-x).^2)) / 2 ;
% Wgrad = lambda * (W' * W * x./sqrt((W*x).^2+epsilon))+ W * (W' * W * x - x) * x' + W * x * (W' * W * x - x)';
Wgrad = lambda * (W*x./sqrt((W*x).^2+epsilon))*x'+ W * (W' * W * x - x) * x' + W * x * (W' * W * x - x)';

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);