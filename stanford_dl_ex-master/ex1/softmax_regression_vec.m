function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
%   f = 0;
%   g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  A = exp([theta' * X;zeros(1,m)]);
  B = bsxfun(@rdivide, A, sum(A));
  C = log(B);
  I = sub2ind(size(C),y,1:size(C,2)); 
  f = (-1) * sum(C(I));
  
  %%%%%%% calculate g %%%%%%%%%%%%
  Y = repmat(y',1,num_classes);
  for i=1:num_classes
      Y(Y(:,i)~=i,i) = 0;
  end
  Y(Y~=0)=1;
  % ����ȥ��Y��һ�У�B��һ������Ϊthetaֻ��num_classes-1��
  g = (-1) * X * (Y(:,1:(size(Y,2)-1))-B(1:(size(B,1)-1),:)');
  %%% ���˵�д��������д��Ч��һ��,��Ҫ��ϡ��������ɲ�һ��һ�㣬�����ٶ��Կ�%%
  %%% ��Ϊ����num_classes����С���Һ�ʱ0.014272�룬���ĺ�ʱ0.014249�� %%%
%   h = theta'*X;%h(k,i)��k��theta����i������
%   a = exp(h);
%   a = [a;ones(1,size(a,2))];%��1��
%   p = bsxfun(@rdivide,a,sum(a));
%   c = log2(p);
%   i = sub2ind(size(c), y,[1:size(c,2)]);
%   values = c(i);
%   f = -sum(values);

%   d = full(sparse(1:m,y,1));
%   d = d(:,1:(size(d,2)-1));%��1��
%   p = p(1:(size(p,1)-1),:);%��1��
%   g = X*(p'.-d);

  g=g(:); % make gradient a vector for minFunc

