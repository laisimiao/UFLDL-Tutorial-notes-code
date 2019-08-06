function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%   [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
%   SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);  %here,theta is opt_params,stack size:(2,1),stack include network parameters
numHidden = numel(ei.layer_sizes) - 1; % num of hidden layers,here numHidden=1
hAct = cell(numHidden+1, 1); % hAct record each layer input and output(except input layer)
gradStack = cell(numHidden+1, 1); % gradStack record gradient of backprop
m = size(data,2); % number of examples
%% forward prop
%%% YOUR CODE HERE %%%
for l = 1:numHidden+1
    if(l == 1)
        hAct{l}.z = stack{l}.W*data;  % 第一个隐层，将训练数据集作为其数据
    else
        hAct{l}.z = stack{l}.W*hAct{l-1}.a; % 第l层的输入，是第l-1层的输出，
    end
    hAct{l}.z = bsxfun(@plus,hAct{l}.z,stack{l}.b); % 第l层的节点的输入加上偏置
    hAct{l}.a = sigmoid(hAct{l}.z); % 应用激活函数
end

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  mat_e = exp(hAct{numHidden+1}.z); 
  pred_prob = bsxfun(@rdivide,mat_e,sum(mat_e,1));
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
mat_e = exp(hAct{numHidden+1}.z); % 而网页上是theta * hw,b(x),就是经过激活的a size:(10,60000)
pred_prob = bsxfun(@rdivide,mat_e,sum(mat_e,1));
I = sub2ind(size(pred_prob),labels',1:size(pred_prob,2));
ceCost = -sum(log(pred_prob(I)));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
tabels = zeros(size(pred_prob));
tabels(I) = 1;
for l = numHidden+1:-1:1 
    if(l == numHidden+1)
        hAct{l}.delta = -(tabels - pred_prob);  % 输出层使用softmax的损失函数，所以和二次项损失函数不同，其他的都是一样的
    else
        hAct{l}.delta = (stack{l+1}.W'* hAct{l+1}.delta) .* (hAct{l}.a .*(1- hAct{l}.a));
    end
    
    if(l == 1)
        gradStack{l}.W = hAct{l}.delta*data'; %hAct{0}.a相当于输入data
        gradStack{l}.b = sum(hAct{l}.delta,2);
    else
        gradStack{l}.W = hAct{l}.delta*hAct{l-1}.a';
        gradStack{l}.b = sum(hAct{l}.delta,2);        
    end    
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for l = 1:numHidden+1
    wCost = wCost+ sum(sum(stack{l}.W.^2));   %  网络参数W的累计和,正则项损失
end
cost = (1/m)*ceCost + .5 * ei.lambda * wCost; % 带正则项的损失

% Computing the gradient of the weight decay.
for l = numHidden+1: -1 : 1
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



