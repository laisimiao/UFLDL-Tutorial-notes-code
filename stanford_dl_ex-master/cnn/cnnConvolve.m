function convolvedFeatures = cnnConvolve(filterDim, numFilters, images, W, b)
% convolvedFeatures = cnnConvolve(filterDim, numFilters, convImages,   W,     b);
% in cnnExercise.m                   8          100       28*28*8   8*8*100 100*100
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numImages = size(images, 3);
imageDim = size(images, 1);  % 方阵
convDim = imageDim - filterDim + 1; % 28 - 8 + 1 = 21

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

% Instructions:
%   Convolve every filter with every image here to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 30 seconds 
%   Convolving with 5000 images should take around 2 minutes
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)


for imageNum = 1:numImages
  for filterNum = 1:numFilters

    % convolution of image with feature matrix
    convolvedImage = zeros(convDim, convDim);

    % Obtain the feature (filterDim x filterDim) needed during the convolution

    %%% YOUR CODE HERE %%%
    filter = squeeze(W(:,:,filterNum));
    % Flip the feature matrix because of the definition of convolution, as explained later
    filter = rot90(squeeze(filter),2);  % squeeze 删除单一维度 二维数组不受 squeeze 的影响
      
    % Obtain the image
    im = squeeze(images(:, :, imageNum));

    % Convolve "filter" with "im", adding the result to convolvedImage
    % be sure to do a 'valid' convolution

    %%% YOUR CODE HERE %%%
    convolvedImage = conv2(im,filter,'valid'); % 21*21
    % Add the bias unit
    % Then, apply the sigmoid function to get the hidden activation
    
    %%% YOUR CODE HERE %%%
    convolvedImage = convolvedImage + b(filterNum);
    convolvedImage = sigmoid(convolvedImage);
    
    convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end

%%%%%%%%%%%%%%%%%%% use gpu(can comment) %%%%%%%%%%%%%
% for imageNum = 1:numImages
%   for filterNum = 1:numFilters
% 
%     % convolution of image with feature matrix
%     convolvedImage = zeros(convDim, convDim);
%     gpu_convolvedImage = gpuArray(convolvedImage);
% 
%     % Obtain the feature (filterDim x filterDim) needed during the convolution
% 
%     %%% YOUR CODE HERE %%%
%     filter = squeeze(W(:,:,filterNum));
%     % Flip the feature matrix because of the definition of convolution, as explained later
%     filter = rot90(squeeze(filter),2);  % squeeze 删除单一维度 二维数组不受 squeeze 的影响
%       
%     % Obtain the image
%     im = squeeze(images(:, :, imageNum));
% 
%     % Convolve "filter" with "im", adding the result to convolvedImage
%     % be sure to do a 'valid' convolution
% 
%     %%% YOUR CODE HERE %%%
%     gpu_filter = gpuArray(filter);
%     gpu_im = gpuArray(im);
%     gpu_convolvedImage = conv2(gpu_im,gpu_filter,'valid');
%     % Add the bias unit
%     % Then, apply the sigmoid function to get the hidden activation
%     
%     %%% YOUR CODE HERE %%%
%     convolvedImage = gpu_convolvedImage + b(filterNum);
%     convolvedImage = sigmoid(convolvedImage);
%     
%     convolvedFeatures(:, :, filterNum, imageNum) = gather(convolvedImage);
%   end
% end

end

