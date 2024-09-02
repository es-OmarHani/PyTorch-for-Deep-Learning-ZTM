# PyTorch Projects and Tutorials

This repository contains a collection of PyTorch projects and tutorials designed to help you understand and apply various concepts in machine learning and deep learning using PyTorch. Each notebook is organized to build on the previous one, covering a range of topics from basic tensor operations to advanced model architectures and experiment tracking.

## Table of Contents

1. [PyTorch Workflow](#01_pytorch_workflow)
2. [PyTorch Classification](#02_pytorch_classification)
3. [PyTorch Computer Vision](#03_pytorch_computer_vision)
4. [PyTorch Custom Datasets](#04_pytorch_custom_datasets)
5. [PyTorch Going Modular](#05_pytorch_going_modular)
6. [PyTorch Transfer Learning](#06_pytorch_transfer_learning)
7. [PyTorch Experiment Tracking](#07_pytorch_experiment_tracking)
8. [PyTorch Paper Replicating](#08_pytorch_paper_replicating)

## 01_pytorch_workflow.ipynb

### Description

This notebook introduces the fundamental building blocks of PyTorch, focusing on tensors. You'll learn:

- **Introduction to Tensors**: The core data structure in PyTorch.
- **Creating Tensors**: How to instantiate tensors with various data.
- **Manipulating Tensors**: Operations like addition and multiplication.
- **Tensor Shapes and Indexing**: Handling shape mismatches and indexing multi-dimensional tensors.
- **Mixing PyTorch and NumPy**: Converting between PyTorch tensors and NumPy arrays.
- **Reproducibility**: Ensuring consistent results in experiments.
- **Running on GPU**: Leveraging GPU acceleration for faster computations.

### Objectives

- Understand tensor operations and manipulation.
- Learn how to use GPU acceleration in PyTorch.

## 02_pytorch_classification.ipynb

### Description

This notebook builds on the PyTorch workflow to tackle a classification problem. You'll explore:

- **Architecture of Classification Networks**: Basic structure of neural networks for classification.
- **Data Preparation**: Creating and processing binary classification datasets.
- **Building a Classification Model**: Defining a model, loss function, optimizer, and training loop.
- **Evaluation and Improvement**: Assessing model performance and making enhancements.
- **Handling Non-linearity**: Incorporating non-linear functions to model complex patterns.

### Objectives

- Apply PyTorch workflow to classification problems.
- Develop and improve classification models.

## 03_pytorch_computer_vision.ipynb

### Description

Applying the PyTorch workflow to computer vision tasks, this notebook covers:

- **Computer Vision Libraries**: Exploring PyTorch's built-in tools for computer vision.
- **Data Loading and Preparation**: Handling image data from FashionMNIST.
- **Baseline Model**: Creating and evaluating a simple classification model.
- **Advanced Models**: Adding non-linearity and building Convolutional Neural Networks (CNNs).
- **Model Comparison**: Evaluating and comparing different models.
- **Confusion Matrix**: Analyzing classification results.

### Objectives

- Implement computer vision tasks using PyTorch.
- Build and evaluate various computer vision models.

## 04_pytorch_custom_datasets.ipynb

### Description

In this notebook, you'll work with a custom dataset of food images to:

- **Import PyTorch and Setup**: Prepare for using custom datasets.
- **Data Preparation**: Understand and transform your own dataset.
- **Custom Dataset Class**: Load data using a custom `Dataset` class.
- **Data Augmentation**: Enhance dataset diversity with augmentation techniques.
- **Model Building and Evaluation**: Create models, train, and evaluate on custom data.

### Objectives

- Load and preprocess custom datasets.
- Build and train models with non-standard data.

## 05_pytorch_going_modular.ipynb

### Description

This notebook focuses on modularizing your PyTorch code for better scalability and maintainability:

- **Modular Code**: Organizing code into scripts and modules for larger projects.
- **Best Practices**: Techniques for making code reusable and efficient.

### Objectives

- Learn to modularize PyTorch code.
- Understand best practices for maintaining large-scale projects.

## 06_pytorch_transfer_learning.ipynb

### Description

Explore transfer learning by adapting pre-trained models:

- **Setup and Data Preparation**: Prepare data and reuse previous code.
- **Customizing Pre-trained Models**: Modify models from `torchvision.models`.
- **Training and Evaluation**: Fine-tune and assess pre-trained models on your data.

### Objectives

- Apply transfer learning to enhance model performance.
- Customize pre-trained models for specific tasks.

## 07_pytorch_experiment_tracking.ipynb

### Description

Track and manage experiments using TensorBoard:

- **Setup and Data Preparation**: Initialize TensorBoard and prepare data.
- **Training with Tracking**: Monitor training progress with TensorBoard.
- **Experiment Management**: Run and compare multiple experiments.

### Objectives

- Use TensorBoard for experiment tracking.
- Create and manage multiple experiments effectively.

## 08_pytorch_paper_replicating.ipynb

### Description

Replicate the Vision Transformer (ViT) architecture from a research paper:

- **Setup and Data Preparation**: Prepare data and initialize the project.
- **Replicating the ViT Model**: Implement components of the Vision Transformer.
- **Training and Evaluation**: Train and evaluate the Vision Transformer on custom data.

### Objectives

- Replicate advanced neural network architectures.
- Apply research techniques to practical problems.

## Going Modular

Add a folder with modular files for better organization and reuse of code across projects.

Feel free to explore each notebook to learn and experiment with different PyTorch concepts and techniques!

