# CNN part

# 1. introduction

In order to tackle multi-label image classification task, we design a Convolutional Neural Network (CNN) . This CNN architecture borrows ideas from Inception, ResNet, VGG, combing data augmentation and category balancing technology, offering flexibility and reliability on classification tasks. Specifically, our model combines the multi-scale feature extraction capability of the Inception module with the residual connection advantage of ResNet. Through this way, we could expect that retain shallow features and learn deep features at the same time. Therefore, it could lead to  improvement on the metrics of multi-label prediction(mdpi.com). At the same time, we introduce data augmentation strategies such as random flipping and cropping to expand the effective training sample space. To some extent,  it could alleviate the problem of  umbalancing categories by oversampling minority category samples.

# 2. related works

**Overview methods of multi-label image classification task:** In recent years, it has been proved effective and efficient of deep convolutional neural network when refers to multi-label image classification task. In the early stage, the AlexNet etc. try to increasing data size and model depth to realize breakthroughs on performance. Then followed by VGG, providing small kernels ideas which is now still commonly used in CNNs. However, when the depths get deeper, the vanishing gradient problem has become significant. ResNet (Residual Network) , which was proposed by He Kaiming in 2015, demonstrating the residual connections could alleviate the degradation problem of deep networks. This has profound impact on later's neural network architecture and become the foundation of current convolutional neural network. This could be explained that by adding shortcut paths of identity mapping between network layers, ResNet can train extremely deep networks such as 50 or 101 layers without serious performance degradation. The residual structure allows the gradient to pass some layers and propagate directly, greatly improving the training stability and accuracy of deep networks. Furthermore, the Inception series network proposed by Google, using different sizes of convolution kernels to extract multi-scale features in parallel inside the Inception module, which improves parameter utilization efficiency and performance. For example, InceptionV3 obtains information of different scales through parallel 1×1, 3×3, and 5×5 convolution branches while controlling computational costs. In 2016, Szegedy et al. proposed to merge Inception with ResNet to form the Inception-ResNet architecture. Inception-ResNet-v1 and v2 add residual connections to each Inception module based on Inception-v4. These advanced CNN architectures provide a powerful feature extraction foundation for multi-label classification.

**Data augmentation**: Krizhevsky et al. introduced random cropping, horizontal flipping, and color perturbation to expand the training data in ImageNet classification. Generally, applying random but label-invariant transformations, such as rotation, translation, color shift to existing images could amplify the distribution of training samples, making the model more adaptable to the test data. Therefore, in some cases, it could make up for the imbalanced data problem. 

**Imbalanced data processing**: In some pratical cases, some labels have far fewer samples than other labels. If trained directly without any processing, the model tends to predict the majority class with more samples, resulting in extremely low recall rate for the minority class. Oversampling minority class samples and undersampling majority class samples are widely used. On top of that, we could give higher weights on minority class and lower weights on majority class. Through these ways, we could make our model more reliable and stable. 

# 3. Techniques

## 3.1 data preprocessing and augmentation

**data preprocessing**: according to observation, the size of most images in this experiment is within 320x320. Hence, we could use zero padding or scaling to a uniform size of 320×320 pixels. This way can align all inputs to a fixed size without cropping the content, facilitating batch training and leveraging pre-trained models.

**Data augmentation**: based on the theory that data augmentation can add random perturbations, which reduces the model's dependence on certain backgrounds or framing biases, thereby performing better on the validation and test dataset. In this experiment,  we introduce that: 

(1) Random horizontal flipping: mirroring the image with a probability of 50% helps the model adapt to changes in left and right viewing angles.

(2) random rotation of a certain angle (for example, ±15°) makes the model insensitive to changes in the object's orientation.

(3) color jitter: randomly changing the brightness, contrast, and hue of the image to make the model adapt to different lighting and color conditions.

(4) random cropping: randomly cropping a sub-region of size close to 320×320 from the image and enlarging it, which is equivalent to simulating the translation or zoom of the camera's field of view.

During these processes, we need to keep the image labels unchanged to ensure that the model learns a robust representation of the content. After data augmentation, the effective sample size of the training set is greatly increased, alleviating the risk of overfitting.

**Category Balancing Strategy**: To address the category imbalance in the training data, we used oversampling to balance the number of samples for each label. To be precise, for labels with fewer samples, we generate more samples by repeatedly sampling their related images and applying the above random enhancements to these images until the number of samples for each target minority class increases to at least about 2500~3000. Through this augmentation-based oversampling method, we make the distribution of each label in the training set more balanced, reducing the bias towards the majority class when training  model. 

## 3.2 CNN architecture

**Structure:** we design a neural network which is the combination of Inception series network and the ResNet residual idea. It is proposed by Szegedy et al., which adds residual connections to each Inception module based on Inception-v4 to speed up training and improve accuracy. This architecture consists of three stages of Inception-ResNet modules (respectively called A, B, and C modules), with two dimension reduction Reduction modules inserted in between, as well as the Stem module at the beginning of the network and the fully connected layer at the end of the network. The image input pass through a series of convolution and pooling operations in the Stem module, reducing the size from 320×320 to a smaller feature map size 37×37, and then passes through multiple layers of Inception-ResNet-A module, Reduction-A module, Inception-ResNet-B module, Reduction-B module, and Inception-ResNet-C module in sequence. Finally, the network outputs the prediction score of each label through global average pooling and a fully connected layer.

**Stem module:** The Stem module is the initial sequence of convolutional layers of the network, which is used to extract low-level features and reduce the size of the input image. A typical Stem implementation includes several layers of 3×3 convolutions with stride 2 and pooling operations to gradually downsample the input image and compress the information. For example, the Stem of Inception-ResNet-v1 first performs a 3×3 convolution with a stride of 2 on the input, and then after several steps of convolution + pooling, it reduces the image from 320×320 to a 37×37 feature map and increases the number of channels to a certain dimension. Therefore, in this stage, the module compresses the information dense and provides rich basic features for subsequent Inception units.

**Inception-ResNet module:** in this part, there are 3 types of Inception-ResNet module: A, B, C. This is because they will be responsible for processing feature maps of different sizes at different depths of the network. Each Inception-ResNet module has multiple parallel convolution branches inside, for instance, The Inception-ResNet-A module typically contains three branches: 

a 1×1 convolution branch, 

a 3×3 convolution branch, 

and a two-layer 3×3 convolution stack branch.

 The features output by these branches are concatenated in the channel dimension and then filter expanded by a 1×1 convolution. This 1×1 convolution does not have an activation function, and its function is to adjust the number of channels of the module output to match the number of input channels. After matching the channel dimension, the module input will be added to the output transformed by the Inception branch to form a residual connection because there is a strict constraint of identical dimension and size when adding residual shortcuts. Similarly, Inception-ResNet-B and C get this design as well, however,  the size and number of convolution branches are different , for example, the B module is adapted to 18×18 medium-sized feature maps, and the C module is adapted to 8×8 smaller feature maps. The specific convolution kernel size combination of each module is slightly different. Each module uses residual connections to add the feature increments extracted by the branches back to the input features, allowing information to transmit efficiently in the network.

**Reduction module:** The Reduction-A and Reduction-B modules are used to reduce the spatial size of feature maps also known as downsampling, and increase the number of channels at different depths of the network. It is similar to the pooling layer but implemented through convolution. The Reduction module is essentially a special Inception module: it contains multiple parallel branches, at least one of which uses convolution or pooling with a stride of 2 to reduce the width and height. For example, the Reduction-A module may have a 3×3 stride 2 convolution branch, a two-layer 3×3 convolution stack branch, and a stride 2 max pooling branch. The outputs of each branch are concatenated in the channel dimension to obtain output features with half the size but increased number of channels. As for the Reduction-B, the module performs a second downsampling, with a similar structure but different parameter scale. The network uses the Reduction module to reduce the feature map from the initial 37×37 to 18×18 and then to 8×8, thereby gradually extracting higher-level abstract features. In this module, we could downsample the model like maxpooling layer and gain more low-frequency information as well like inception module. Hence, it is  effective to add this module into neural network architecture. 

**Residual connection design**:  Inside each Inception-ResNet module, we directly add the output of the convolution transformation to form the output of the module. This means What the network learns is the residual part of the input that needs to be adjusted, while the original feature x can be directly passed through a shortcut. This could be explained that after addition, the nonlinear result of y is generally output through activation functions such as ReLU. 
$$
y=f(x)+x
$$
Another advantage of residual connection is that gradients can be propagated along shortcuts without attenuation, significantly alleviating the gradient vanishing problem of deep networks. At a deeper level, the identity mapping branch provides an information highway that enables the network to retain shallow features for subsequent decision-making, while superimposing and learning more advanced residual features. This ensures that even if many layers are added, the network performance will not degrade, because those layers can at least learn the identity mapping without harming the original information. In addition, residual connections do not add additional parameters and computational complexity, but it improves the training effect and final accuracy of the network.

## 3.3 Multi-label classification loss function and training

**Multi-label prediction output**: For multi-label tasks, the last layer of the model uses Sigmoid activation to obtain independent probability estimates for each label instead of Softmax. Specifically, we let the network output a real-valued score for each possible label and map it to [0, 1].
$$
\hat{y}_i = \sigma(z_i)
$$
**BCE Loss**: We choose the binary cross-entropy loss function (BCE) as the objective function for multi-label learning. For each training sample, the loss calculates the cross entropy independently on each label and then takes the average:
$$
L=− 
N
1
​
  
i=1
∑
N
​
 [y 
i
​
 log 
y
^
​
  
i
​
 +(1−y 
i
​
 )log(1− 
y
^
​
  
i
​
 )]
$$
The reason for using BCE is that the prediction of each label is regarded as an independent binary classification problem, and the calculation of the loss does not affect each other.

**Training configuration and implementation details**: we use typorch to build convolutional neural network and choose He initialization. During training, we choose the Adam optimizer to accelerate convergence, and set the batch size to 32. 
$$
\alpha=0.001, momentum\beta=0.9
$$
The model is trained for several epochs, each of which contains an iteration of the entire training set. Due to the use of strong data augmentation and residual networks, we observed stable convergence and no obvious signs of overfitting. The validation set is used to monitor the model performance during training, and training is stopped early when the validation loss is no longer reduced to avoid overfitting.

# 4. Experiments and Results

**Datasets and evaluation metrics**: 

**Hyperparameters:**

Epochs：20

Optimizer：Adam (Learning rate = 0.001)

Batch size：32

Loss Function：Binary Cross Entropy (BCE Loss)

**Overall Results**: 

| 模型（Model）               | Micro-F1（%） |
| --------------------------- | ------------- |
| Baseline（无数据增强）      | 65.10%        |
| 类别加权（无数据增强）      | 68.21%        |
| Data Augmentation（本研究） | **74.45%**    |

By introducing category weights and data augmentation strategies, the Micro-F1 index of the model is significantly improved. Among them, the model performance is slightly improved when only category weights are added, and the performance is significantly improved after further adding data augmentation strategies, with the Micro-F1 score reaching the highest 74.45%.

## Ablation Studies

### （1）数据增强的效果分析（Effectiveness of Data Augmentation）

The data augmentation strategy aims to increase sample diversity and reduce the risk of model overfitting by random rotation, flipping, color jittering, etc. Figure 1 shows the model performance (Baseline model) when the data augmentation strategy is not adopted, and Figure 2 shows the results of adding category weights but not performing data augmentation.

从图1: baseline

- The baseline model without data augmentation has generally low F1 scores in all categories, especially the categories with fewer samples (such as Class 14, Class 15, and Class 18), which perform the worst, with F1 even close to 0.

从图2（类别加权，无数据增强）：

- After introducing category weights, the F1 scores of minority categories are significantly improved, which shows that category weighting has alleviated the category imbalance problem to a certain extent, but the overall performance improvement is relatively limited.

After adding data augmentation (Figure 3, i.e. the method in this paper), the overall Micro-F1 score of the model is greatly improved to 74.45%, significantly exceeding the 65.10% without augmentation and the 68.21% with weighting only. This significant performance improvement proves that the data augmentation strategy effectively improves the generalization ability and performance stability of the model.

### （2）类别权重对少数类别的影响分析（Impact of Class Weights）

When the class imbalance problem is prominent, the model tends to be biased towards the majority class samples. By giving a larger weight to the minority class in the loss function, the model can pay more attention to the minority class and improve its recall rate and F1-score performance.

Specifically, the model without data enhancement but with weights introduced in Figure 2 has significantly improved F1 scores on minority classes such as Class 14, Class 15, and Class 18 compared to the Baseline (Figure 1). This result clearly shows that class weighting is a simple and effective method that can improve the performance deviation caused by class imbalance in a targeted manner.

# 5. Conclusion and discussion

This paper proposes and implements an effective deep learning scheme for multi-label image classification tasks. By designing  the Inception-ResNet-v1 deep residual network, our model can efficiently extract multi-scale features of images and use residual connections to retain low-level information, successfully coping with the complex situation of multiple targets or attributes in one image. The experimental results verify the effectiveness of deep networks combined with data enhancement: compared with shallow models, Inception-ResNet significantly improves the average F1, showing its advantage in capturing rich features. 

At the same time, data enhancement strategies such as random rotation and flipping significantly improve the generalization ability of the model, enabling it to maintain robust performance on unseen data. In addition, by augmenting and balancing the minority categories, we alleviate the imbalance problem of training data, thereby achieving more balanced precision and recall on each label.

 To sum up, the method of this study has achieved satisfactory results in multi-label image classification, proving the necessity and effectiveness of adopting deep residual architecture + data enhancement + category balance.