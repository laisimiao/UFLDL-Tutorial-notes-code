function [train, test] = ex1_load_mnist(binary_digits)

  % Load the training data
  X=loadMNISTImages('E:\SummerCourse\UFLDL\common\train-images-idx3-ubyte');
  y=loadMNISTLabels('E:\SummerCourse\UFLDL\common\train-labels-idx1-ubyte')'; % 这个括号后面的“'”不能删，否则第10行出错

  if (binary_digits)
    % Take only the 0 and 1 digits
    X = [ X(:,y==0), X(:,y==1) ]; % y==0返回的是逻辑向量，等于0的地方附逻辑1，非0的地方附逻辑0
    y = [ y(y==0), y(y==1) ];
  end

  % Randomly shuffle the data
  I = randperm(length(y)); % 12665
  y=y(I); % labels in range 1 to 10
  X=X(:,I);

  % We standardize the data so that each pixel will have roughly zero mean and unit variance.
  s=std(X,[],2);
  m=mean(X,2);
  X=bsxfun(@minus, X, m);
  X=bsxfun(@rdivide, X, s+.1);

  % Place these in the training set
  train.X = X;
  train.y = y;

  % Load the testing data
  X=loadMNISTImages('E:\SummerCourse\UFLDL\common\t10k-images-idx3-ubyte');
  y=loadMNISTLabels('E:\SummerCourse\UFLDL\common\t10k-labels-idx1-ubyte')';

  if (binary_digits)
    % Take only the 0 and 1 digits
    X = [ X(:,y==0), X(:,y==1) ];
    y = [ y(y==0), y(y==1) ];
  end

  % Randomly shuffle the data
  I = randperm(length(y));
  y=y(I); % labels in range 1 to 10
  X=X(:,I);

  % Standardize using the same mean and scale as the training data.
  X=bsxfun(@minus, X, m);
  X=bsxfun(@rdivide, X, s+.1);

  % Place these in the testing set
  test.X=X;
  test.y=y;