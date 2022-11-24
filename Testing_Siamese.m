imdsTest = imageDatastore("data/test", "IncludeSubfolders",true,"LabelSource","foldernames");


files = imdsTest.Files;
parts = split(files,filesep);
labels = join(parts(:,(end-2):(end-1)),"_");
imdsTest.Labels = categorical(labels);

numberClasses = numel(unique(imdsTest.Labels))

accuracy = zeros(1,5);
batchSizeAccuracy = 10;

for i = 1:5
    % mini-batch of image pairs and pair labels is extracted.
    [X1,X2,pairsLabelsAcc] = getSiameseBatch(imdsTest,batchSizeAccuracy);

    % Mini-batch data conversion to dlarray. 
    % "SSCB" (spatial, spatial, channel, batch) for image data processing.
    X1 = dlarray(X1,"SSCB");
    X2 = dlarray(X2,"SSCB");

    % Convert data to gpuArray if a GPU is available.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        X1 = gpuArray(X1);
        X2 = gpuArray(X2);
    end

    % Using a trained network, evaluate predictions
    Y = predictionMake(net,fcParams,X1,X2);

    % Convert predictions to binary 0 or 1 values.
    Y = gather(extractdata(Y));
    Y = round(Y);

    % Calculate the average accuracy for the minibatch.
    accuracy(i) = sum(Y == pairsLabelsAcc)/batchSizeAccuracy;
end

averageAccuracy = mean(accuracy)*100

function Y = predictionMake(net,fcParams,X1,X2)

% Pass the first image through the siamese subnetwork..
Y1 = predict(net,X1);
Y1 = sigmoid(Y1);

% Pass the second image through the siamese subnetwork..
Y2 = predict(net,X2);
Y2 = sigmoid(Y2);

% Subtract the feature vectors.
Y = abs(Y1 - Y2);

Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);

Y = sigmoid(Y);

end

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
