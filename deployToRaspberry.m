%raspberry pi connection to matlab
mypi = raspi();

%default webcam raspberry pi 
cam1 = webcam(mypi,1);

%microsoft webcam USB connection to raspberry pi
cam2 = webcam(mypi,2);

%Initializing network and image input size
net = coder.loadDeepLearningNetwork("net");
inputSize = [150, 150, 1];

faceDetection = vision.CascadeObjectDetector();

while true

    %capture image from default webcam
    img1 = snapshot(cam1);
    bbox = step(faceDetection,img1);
    img1 = insertObjectAnnotation(img1,"rectangle", bbox,"Face");
    imshow(img1);
    %delay for 10 seconds
    pause(10);

    %capture image from microsoft webcam
    img2 = snapshot(cam2);
    bbox = step(faceDetection,img2);
    img2 = insertObjectAnnotation(img2,"rectangle", bbox,"Face");
    imshow(img2);
    %delay for 5 seconds
    pause(5);

    % resize input image
    imgSizeAdjustment = imresize(img1,img2,inputSize(1:2));

    %classify input image
    [prediction, score] = net.classify(imgSizeAdjustment);
    maxScore = max(score);

    predictStr = cellstr(prediction);
    textToDisplay = printf("Prediction : %s \nScore : %f", predictStr, maxScore);

    %Display the prediction
    img_pred = insetText(img2,img1,[0,0], textToDisplay);
    displayImage(mypi,img_pred)

end



