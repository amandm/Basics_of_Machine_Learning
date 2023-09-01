# README

# Implementation of common Algorithms --

Prime device - Mac M2, but can work on any device with slight modification

Recap of useful algorithms related to machine learning, Mostly TF2.0 is used

[1. Simple Neural network for Linear regression](Implimentation%20of%20Common%20Algorithms/1.%20Simple%20Neural%20network.ipynb)

[2. Gradient Decent and SGD from Scratch](Implimentation%20of%20Common%20Algorithms/2.%20Gradient%20Decent%20and%20SGD%20from%20Scratch.ipynb)

[3. Animal Emotion classifier with TensorBoard](Implimentation%20of%20Common%20Algorithms/3.%20Animal%20Emotion%20classifier%20with%20tensorBoard.ipynb)

- Used Pytorch with mps
    neural network-based image classifier using PyTorch, specifically utilizing the ResNet-18 architecture. The focus was on enhancing the training loop by incorporating TensorBoard for real-time monitoring of various metrics such as loss and accuracy for both training and validation phases.
    1. **Running Loss**: The code was enhanced to calculate a running loss for both training and validation, offering an average loss per epoch.
    2. **Epoch Accuracy**: Along with loss, epoch-level accuracy metrics were also calculated for both training and validation datasets.
    3. **TensorBoard Integration**: TensorBoard was utilized to visualize the metrics in real-time, allowing for better interpretability and easier debugging.
    4. **Housekeeping**: Guidance was provided for managing TensorBoard runs, including deleting old runs and killing the TensorBoard process to free up ports.
    ![Untitled](images/Animal%20Emotion%20classifier%20with%20TensorBoard1.png)
    
    ![Untitled](images/Animal%20Emotion%20classifier%20with%20TensorBoard2.jpg)
    

- [ ]  Custom loss function
- [ ]  Recipe to train neural network
- [ ]  Skip connection
- [ ]  Normalization and Batch-Norm
- [ ]  Mutual Information
- [ ]  LSTM and GRU
- [ ]  Attention

## ML Theory Recap

1. [Evaluation metrics - Precision, Recall, F1 Score, ROC, Effect on data imbalance](ML%20Theory%20Recap/Precision,%20Recall%20and%20more.md)
    
    A refresher on Machine Learning (ML) concepts related to evaluation metrics such as Precision, Recall, and F1 score. It includes a breakdown of these metrics using a confusion matrix with labels like True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
    
- [ ]  Batch Normalization
- [ ]  L1 vs L2
- [ ]  Dropout