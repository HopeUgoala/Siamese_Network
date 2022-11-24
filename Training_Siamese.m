% Unzip the dataset
%exampleFiles = unzip('10-shotData.zip')
imdsTrain = imageDatastore("data/training", "IncludeSubfolders",true,"LabelSource","foldernames");


files = imdsTrain.Files;
parts = split(files,filesep);
labels = join(parts(:,(end-2):(end-1)),"-");
imdsTrain.Labels = categorical(labels);

 %{ 
idx = randperm(numel(imdsTrain.Files),8);

for i = 1:numel(idx)
    subplot(4,2,i)
    imshow(readimage(imdsTrain,idx(i)))
    title(imdsTrain.Labels(idx(i)),Interpreter="none");
end
 %} 
 %{ 
batchSize = 10;
[pairImage1,pairImage2,pairLabel] = getSiameseBatch(imdsTrain,batchSize);

for i = 1:batchSize
    if pairLabel(i) == 1
        s = "similar";
    else
        s = "dissimilar";
    end
    subplot(2,5,i)
    imshow([pairImage1(:,:,:,i) pairImage2(:,:,:,i)]);
    title(s)
end

 %} 


layers = [
    imageInputLayer([150 150 1],Normalization="none")
    convolution2dLayer(10,64,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(8,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(4,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(4,256,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    fullyConnectedLayer(4096,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")];

lgraph = layerGraph(layers);

figure
plot(lgraph);
net = dlnetwork(lgraph);


fcWeights = dlarray(0.01*randn(1,4096));
fcBias = dlarray(0.01*randn(1,1));

fcParams = struct(...
    "FcWeights",fcWeights,...
    "FcBias",fcBias);


numIterations = 1500;
miniBatchSize = 128;

learningRate = 2e-5;
gradDecay = 0.9;
gradDecaySq = 0.99;

executionEnvironment = "auto";

figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

trailingAvgSubnet = [];
trailingAvgSqSubnet = [];
trailingAvgParams = [];
trailingAvgSqParams = [];


start = tic;

% Loop over mini-batches.
for iteration = 1:numIterations

    % Mini-batch extraction of image pairs and pair labels.
    [X1,X2,pairsLabels] = getSiameseBatch(imdsTrain,miniBatchSize);

    % Convert mini-batch of data to dlarray. Specify the dimension labels
    X1 = dlarray(X1,"SSCB");
    X2 = dlarray(X2,"SSCB");

    % Convert data to gpuArray if training on a GPU..
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        X1 = gpuArray(X1);
        X2 = gpuArray(X2);
    end

    % Evaluate the model loss and gradients with dlfeval and the 
    % modelLoss function listed at the end of the example.
    [loss,gradientsSubnet,gradientsParams] = dlfeval(@modelLoss,net,fcParams,X1,X2,pairsLabels);

    % Update the Siamese subnetwork parameters.
    [net,trailingAvgSubnet,trailingAvgSqSubnet] = adamupdate(net,gradientsSubnet, ...
        trailingAvgSubnet,trailingAvgSqSubnet,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the fullyconnect parameters.
    [fcParams,trailingAvgParams,trailingAvgSqParams] = adamupdate(fcParams,gradientsParams, ...
        trailingAvgParams,trailingAvgSqParams,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the training loss progress plot.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    lossValue = double(loss);
    addpoints(lineLossTrain,iteration,lossValue);
    title("Elapsed: " + string(D))
    drawnow
end

disp(lossValue)
save('face_siameseNetwork.mat', 'trailingAvgSqParams', 'trailingAvgParams','fcParams', 'net')

function [X1,X2,pairsLabels] = getSiameseBatch(imds,miniBatchSize)

pairsLabels = zeros(1,miniBatchSize);
imgSize = size(readimage(imds,1));
X1 = zeros([imgSize 1 miniBatchSize],"single");
X2 = zeros([imgSize 1 miniBatchSize],"single");

for i = 1:miniBatchSize
    choice = rand(1);

    if choice < 0.5
        [pairsIdx1,pairsIdx2,pairsLabels(i)] = getSimilarPairData(imds.Labels);
    else
        [pairsIdx1,pairsIdx2,pairsLabels(i)] = getDissimilarPairData(imds.Labels);
    end

    X1(:,:,:,i) = imds.readimage(pairsIdx1);
    X2(:,:,:,i) = imds.readimage(pairsIdx2);
end

end


function  [pairsIdx1,pairsIdx2,label] = getDissimilarPairData(classLabel)

% Find all of the unique classes.
classes = unique(classLabel);

% Choose two different classes at random to create a dissimilar pair.
classesChoice = randperm(numel(classes),2);

% Determine the indices of all first and second class observations.
idxs1 = find(classLabel==classes(classesChoice(1)));
idxs2 = find(classLabel==classes(classesChoice(2)));

% Choose one image at random from each class.
pairFirstChoice = randi(numel(idxs1));
pairSecondChoice = randi(numel(idxs2));
pairsIdx1 = idxs1(pairFirstChoice);
pairsIdx2 = idxs2(pairSecondChoice);
label = 0;

end


function [pairsIdx1,pairsIdx2,pairLabel] = getSimilarPairData(classLabel)

% Find all distinct classes.
classes = unique(classLabel);

% Choose a class at random that will be used to find a similar pair.
classChoice = randi(numel(classes));

% Determine the indices of all the observations in the selected class.
idxs = find(classLabel==classes(classChoice));

% Choose two images at random from the selected class.
pairIdxChoice = randperm(numel(idxs),2);
pairsIdx1 = idxs(pairIdxChoice(1));
pairsIdx2 = idxs(pairIdxChoice(2));
pairLabel = 1;

end

function [loss,gradientsSubnet,gradientsParams] = modelLoss(net,fcParams,X1,X2,pairsLabels)

% The image pair should be passed through the network.
Y = forwardSiamese(net,fcParams,X1,X2);

% Compute binary cross-entropy loss.
loss = binarycrossentropy(Y,pairsLabels);

% Calculate the loss gradients in relation to the network learnable parameters.
[gradientsSubnet,gradientsParams] = dlgradient(loss,net.Learnables,fcParams);

end

function Y = forwardSiamese(net,fcParams,X1,X2)

Y1 = forward(net,X1);
Y1 = sigmoid(Y1);


Y2 = forward(net,X2);
Y2 = sigmoid(Y2);

% Subtract the feature vectors
Y = abs(Y1 - Y2);

% Run the outcome through a fullyconnect operation.
Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);

Y = sigmoid(Y);

end


function loss = binarycrossentropy(Y,pairsLabels)

% Get prediction precision to avoid errors caused by floating point precision.
precision = underlyingType(Y);

% Convert values with a precision less than floating point precision to eps.
Y(Y < eps(precision)) = eps(precision);
Y(Y > 1 - eps(precision)) = 1 - eps(precision);
loss = -pairsLabels.*log(Y) - (1 - pairsLabels).*log(1 - Y);
loss = sum(loss)/numel(pairsLabels);

end


