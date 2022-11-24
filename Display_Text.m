testBatchSize = 10;
[XTest1,XTest2,pairLabelsTest] = getSiameseBatch(imdsTest,testBatchSize);

XTest1 = dlarray(XTest1,"SSCB");
XTest2 = dlarray(XTest2,"SSCB");

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    XTest1 = gpuArray(XTest1);
    XTest2 = gpuArray(XTest2);
end

YScore = makePrediction(net,fcParams,XTest1,XTest2);
YScore = gather(extractdata(YScore));

YPred = round(YScore);

XTest1 = extractdata(XTest1);
XTest2 = extractdata(XTest2);

f = figure;
tiledlayout(2,5);
f.Position(3) = 2*f.Position(3);

predLabels = categorical(YPred,[0 1],["dissimilar" "similar"]);
targetLabels = categorical(pairLabelsTest,[0 1],["dissimilar","similar"]);

for i = 1:numel(pairLabelsTest)
    nexttile
    imshow([XTest1(:,:,:,i) XTest2(:,:,:,i)]);

    title( ...
        "Target: " + string(targetLabels(i)) + newline + ...
        "Predicted: " + string(predLabels(i)) + newline + ...
        "Score: " + YScore(i))
end

