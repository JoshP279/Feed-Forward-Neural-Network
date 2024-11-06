# Feed Forward Neural Network to Classify Faults in Steel Plates
## Overview

This project demonstrates a neural network-based approach to classify faults in steel plates, leveraging a multilayer neural network trained on 27 real-world attributes of 1,900 steel plates. The goal is to categorize each plate into one of seven fault types, aiming for accurate predictions on a diverse dataset.
Problem Statement
The task involves building a neural network model with one hidden layer, trained using the gradient descent algorithm. This model identifies fault categories by analyzing various plate attributes. The seven fault categories are:

  1. Pastry
  2. Z_Scratch
  3. K_Scatch
  4. Stains
  5. Dirtiness
  6. Bumps
  7. Other_Faults

Each row in the dataset represents a plate with associated attributes, ending with a fault category label. The model's generalization capability was tested on unseen data, including a set of 40 patterns with missing labels for which predictions were required.
Model Configuration and Parameters

  **Input Layer**: 27 neurons (one for each attribute) </br>
  **Hidden Layer**: 35 neurons </br>
  **Output Layer**: 7 neurons (one per fault category) </br>
  **Activation Function**: Sigmoid </br>
  **Learning Rate**: Best results with 0.001 </br>
  **Iterations**: 20,000 </br>

## Results Summary
The network achieved the following accuracies:

  **Training Set**: 85.39% accuracy (1,298 correctly classified out of 1,520) </br>
  **Test Set**: 67.89% accuracy (129 correctly classified out of 190) </br>
  **Final Test Accuracy with Additional Techniques**: 76.32% </br>

## Sum Squared Error (SSE) Metrics:

**Training SSE: 176.96
Test SSE: 40.95**

## Techniques Applied for Performance Enhancement

To optimize training time and generalization, the following methods were investigated and implemented:

* **Appropriate Weight Initialization**: Improved initial training SSE and training stability, though led to faster overfitting if used alone.
* **Noise Injection**: Added slight noise to inputs, improving accuracy by creating a smoother search space.
* **Overfitting Prevention**: Used a validation set to monitor overfitting, with early stopping when validation SSE deviated significantly.

Using these techniques together yielded the best performance, achieving a final test accuracy of 76.32%.
## Observations and Conclusions

The model was influenced by a class imbalance in the dataset, with certain faults (6 and 7) overrepresented, leading to prediction bias. Overfitting prevention and noise injection mitigated this bias to some extent, but considering that this was real world data, a validation accuracy of 76.32% was impressive. The investigation underscored the impact of learning rate, with smaller rates leading to more stable convergence.
