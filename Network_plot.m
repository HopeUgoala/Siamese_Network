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
%plot(lgraph);
net = dlnetwork(lgraph);
analyzeNetwork(net)
