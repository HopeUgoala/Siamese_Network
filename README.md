# Siamese_Network and its application in Face Biometric

![siamense](https://github.com/user-attachments/assets/c1ffc5b0-fdaa-41be-8b91-a4d592938b0b)
#### The setup consists of two identical neural networks, arranged vertically with one above the other. When we input two images into these parallel networks, they produce outputs that are then fed into a Contrastive Loss Function. This function measures the distance or dissimilarity between the two outputs.

![master1](https://github.com/user-attachments/assets/f37f97b5-dd91-488d-8aaa-54d11f6c788c)
#### This CNN subnetwork processes 150x150x1 grayscale images through convolutional, ReLU, and max-pooling layers, progressively reducing spatial size (to 14x14) while increasing channel depth (up to 256). 

![master2](https://github.com/user-attachments/assets/962adffa-046f-4927-b2b6-b90cf1e7d39d)
#### The network goal is to distinguish between the two inputs, Y1 and Y2, as defined in Figure 4.8. The network’s output is a probability between 0 and 1, with a value closer to 0 indicating that the images are dissimilar and a value closer to 1 indicating that the images are similar. The binary cross-entropy between the predicted score and the true label value determines the loss.

![master3](https://github.com/user-attachments/assets/844c1daa-d0a1-405d-8ceb-4623d638cfc7)

![master4](https://github.com/user-attachments/assets/e6af50d6-e19d-4dd0-8115-c87777e5a8de)

![master5](https://github.com/user-attachments/assets/b65340f9-23a5-41f2-9768-2f0e81d8eda7)

![new1](https://github.com/user-attachments/assets/26165118-4922-4efe-a5f7-698f840b01af)

![master7](https://github.com/user-attachments/assets/f6823258-3bcc-4e52-ba10-a63a3ad3a878)

![mater8](https://github.com/user-attachments/assets/0a8f3a9b-ef8c-4f1e-a3e6-bff062a8bcf3)
#### After the Matlab file was deployed to the Raspberry Pi, the two cameras, Microsoft USB Webcam and the other Raspberry Pi Camera, as shown in Figure 4.9 are used to capture images. The result shows that the algorithm was able to distinguish images with a very high confidence score. The result is shown in Figure 5.9.

![master8](https://github.com/user-attachments/assets/8e2ab125-226c-4cc8-94a0-e40529bf70b1)

### Conclusion
#### The dataset was pre-processed first and rescaled to dimensions 150x150x1. A training option was specified for the network, from the training loss progress plots one can see that the model converges after many iterations, as seen in Figure 5.6. When tested with unseen data, the algorithm distinguishes between two images correctly with a high confidence score, as presented in Figure 5.7, and Figure 5.8. Also, the algorithm distinguishes correctly when deployed to Raspberry Pi, as shown in the Figure 5.9.

