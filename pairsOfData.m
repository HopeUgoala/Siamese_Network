% Unzip the dataset
%exampleFiles = unzip('10-shotData.zip')
imdsTrain = imageDatastore("data/training", "IncludeSubfolders",true,"LabelSource","foldernames");


files = imdsTrain.Files;
parts = split(files,filesep);
labels = join(parts(:,(end-2):(end-1)),"-");
imdsTrain.Labels = categorical(labels);

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



