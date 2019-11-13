# Visual-Question-Answering-with-PyTorch
This repository contains the implementations of [Simple Baseline for Visual Question Answering](https://arxiv.org/pdf/1512.02167.pdf) and [Hierarchical Question-Image Co-Attention for Visual Question Answering](https://arxiv.org/pdf/1606.00061.pdf).
## Data Loader  
A dataset loader for VQA (1.0 Real Images).Generally, for mapping the images and questions to the space of Tensors.
Designed for three types of data that the model can process, the image, the question and the classification result. A transformer is tailored for large dataset, to separate the three channels in RGB resizing. Encoding the information as Tensors is also an essential requirement for the use of GPUs, CUDA in specific.

**__ _getitem_ __  dictionary includes:**     
'img' represents the Tensors of the image, 'question' represents the word index, 'target' represents the ground truth

## Simple Baseline  
By combining the cross entropy loss function, the modality of multiple-choice can enable each sample fit multiple labels during training. The cosine similarities are stored, so that the comparison between the picked answers and the labeled answers could be performed. During the actual testing, where the label will be placed remains unknown, therefore fitting multiple labels will increase the accuracy of performance. Adamax optimizer is used.

In order to compute the ground truth answer, I converted the VQA problem into a classification problem by selecting 2185 answers whose answer frequency are greater than or equal to 9. These answers are selected, then are further used for concatenating the word feature extracted from the questions (encoding) with the features extracted from the image.

The word embedding model is implemented to replace the original bag of words (BoW) model. Instead of using the one-hot vector encoding in BoW, the custom encoding with decimal format are used. Thus a large amount of semantic information can be incorporated through training, which is also convenient for the subsequent network processing.

**Accuracy and loss**  
<img src="https://github.com/WeijiaGao49/Visual-Question-Answering-with-PyTorch/raw/master/result/baseline.png" width=60% height=60%>

## Co-Attention Network  
Parallel co-attention mechanism is implemeted for hierarchical question processing, attention, and the use of recurrent layers.
In order to speed up training, a padding with length of four is added in front of the sentence, therefore the final state of LSTM can be used for question features. For every time point in LSTM, the state and output of the hidden layer are considered, while the final state output contains information for the entire question may also represents the overall question. And to incorporate more information, I changed the final judgment of the classifier and concatenated the final output for LSTM with question features and image features.

**Accuracy and loss**  
<img src="https://github.com/WeijiaGao49/Visual-Question-Answering-with-PyTorch/raw/master/result/coattention.png" width=60% height=60%>

## Custom Network
When extracting features from images, the features of the middle layer and the results of the output layer are all taken as image features. Take GoogleNet as an example, by extracting the 1024-dimensional features of the middle layer and the 1000-dimensional features of the output layer, we can fully make use of the capabilities of the pre-trained model of ImageNet. Since the 1000-dimensional classification information of the original ImageNet classification is added, we can know whether similar information appeared for certain objects, which is essential for VQA problems.

Further improvements lie in:

1. Incorporating the image segmentation information into the VQA model. Namely the model can answer questions when knowing the specific content of each region. Instead of extracting the feature of the bounding box within the detected image, by implementing the segmentation for features, an object can be extracted directly, then its surroundings can be treated as background for feature extraction.

2. Using the type of questions as features to help the model understand the problem in more depth. For instance, regarding to the question type of "who", if the model can generate answers for the question type of "how many", further performance enhancement is expected.

**Accuracy and loss**  
<img src="https://github.com/WeijiaGao49/Visual-Question-Answering-with-PyTorch/raw/master/result/custom.png" width=60% height=60%>

## How to run the code
1. Create a ./data folder to store dataset and intermediate data
2. Put the grow.6B.300d.txt file in the ./data/glove folder
3. Run /src/preprocess/create_dictionary.py
Install h5py with the command "python -m pip install h5py", save the file 'OpenEnded_mscoco_train2014_questions.json' under the directory of data/. Run src/preprocess/create_dictionary.py
4. Run /src/preprocess/create_softscore.py
Save the file 'mscoco_train2014_annotations.json' under the directory of data/. Run /src/preprocess/compute_softscore.py. The output 'Num of answers that appear >= 9 times: 2185' indicates that the model has 2185 classifications.
5. Start training with main_co.py
