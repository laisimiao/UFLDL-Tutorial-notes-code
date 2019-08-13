function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%                                   3        21*21*100*8
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages); % 7*7*100*8

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
%% METHOD1:Using mean to pool
% for imageNum = 1:numImages
%   for filterNum = 1:numFilters
%       pooledImage = zeros(convolvedDim / poolDim, convolvedDim / poolDim);
%       im = convolvedFeatures(:,:,filterNum, imageNum);
%       for i=1:(convolvedDim / poolDim)
%           for j=1:(convolvedDim / poolDim)
%               pooledImage(i,j) = mean(mean(im((i-1)*poolDim+1:i*poolDim,(j-1)*poolDim+1:j*poolDim)));
%           end
%       end
%       
%       pooledFeatures(:,:,filterNum, imageNum) = pooledImage;
%   end
% end
%%======================================================================
%% METHOD2:Using conv2 as well to pool
% (if numImages is large,this method may be better,can use "gpuArray.conv2"to speed up!)
pool_filter = 1/(poolDim*poolDim) * ones(poolDim,poolDim);
for imageNum = 1:numImages
  for filterNum = 1:numFilters
      pooledImage = zeros(convolvedDim / poolDim, convolvedDim / poolDim);
      im = convolvedFeatures(:,:,filterNum, imageNum);
      for i=1:(convolvedDim / poolDim)
          for j=1:(convolvedDim / poolDim)
              temp = conv2(im,pool_filter,'valid');
              pooledImage(i,j) = temp(poolDim*(i-1)+1,poolDim*(j-1)+1);
          end
      end
      
      pooledFeatures(:,:,filterNum, imageNum) = pooledImage;
  end
end
end

