# Swift Artificial Neural Network

## Description
This project is an artificial neural network implementation inspired by my good friend and former professor, [Mihai Oltean](https://github.com/mihaioltean), and his [original implementation](https://github.com/mihaioltean/ANN) in C/C++. It was initially started as a university course project back in 2018. However, due to its incompleteness, I decided to rewrite it from scratch and create a fresh and improved artificial neural network in Swift.
The project utilizes the popular [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for training and testing the neural network.


## Features

- Neural network architecture with customizable input, hidden, and output nodes.
- Training the neural network using backpropagation and gradient descent.
- Activation functions: ReLU and softmax.
- Evaluation of the neural network's performance on test data using the evaluate method.
- Prediction of output values for new input data using the predict method.
- Support for saving and loading trained models to/from files as .json.
- Loading of training and testing data from a .txt file. (Supports 2D arrays)

## Prerequisites

- Swift programming language
- Xcode or Swift command-line tools

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/alexdonea/ann-swift
    ```
2. Navigate to the project directory and locate the main.swift file in the project directory and open it in your favourite IDE or editor.
3. Update the file location for project folder:
   ```swift
    let path = "/your-path/ann-swift"
   ```
4. Save the changes and run the `main.swift` file using the Swift compiler or simple run project pressing `CMR + R` in Xcode:
   ```bash
   swift main.swift
   ```
## Dataset Format

The training and testing data for this project is provided in .txt files. You can use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) or any other dataset of your choice. The dataset should be formatted as follows:

- The first line contains information about the model, including the number of training data samples, the number of input variables, and the number of output classes. For example: `60000 784 10`.
- The remaining lines represent individual data samples, where each line corresponds to a single sample.
- Each line consists of input variables followed by the target output. For example `... 0 0.0117647 0.0705882 5`
- The input variables and target output are space-separated values, with the target value placed at the end of the line. For example: `5`

## Usage


```swift
// Create an instance of the model file manager
let modelFileManager = ModelFileManager()

// Path to the 'mnist' folder inside the project
let path = "/your-path/ann-swift"
let trainDataFile = "\(path)/mnist/mnist_train.txt"
let testDataFile = "\(path)/mnist/mnist_test.txt"

// Optional: Specify the file path to load or save the model
let modelFileURL = URL(fileURLWithPath: "\(path)/mnist/model.json")

// Load training and testing data
let (status,
     trainingData,
     trainingTarget,
     testData,
     testTarget,
     trainingNumVariables,
     trainingNumOutputs) = modelFileManager.loadTrainAndTestData(trainLocation: trainDataFile,
                                                                 testLocation: testDataFile)

if status {
    // Set up neural network parameters
    let inputNodes = trainingNumVariables
    let hiddenNodes = 100
    let outputNodes = trainingNumOutputs
    let learningRate = 0.1
    let numEpochs = 100
    let batchSize = 32
    let dropoutRate = 0.2
    
    // Create an instance of the artificial neural network
    let neuralNetwork = ArtificialNeuralNetwork(inputNodes: inputNodes,
                                                hiddenNodes: hiddenNodes,
                                                outputNodes: outputNodes,
                                                learningRate: learningRate)
    
    // Train the neural network using the training dataset
    print("Training started.")
    neuralNetwork.train(trainingData,
                        trainingTarget,
                        batchSize: batchSize,
                        epochs: numEpochs,
                        learningRateDecay: learningRate,
                        dropoutRate: dropoutRate)
    
    // Validate the neural network using the test dataset
    print("Validation started.")
    if let (averageError, accuracy) = neuralNetwork.validate(inputs: testData, targets: testTarget) {
        print("Validation error: \(averageError), Accuracy: \(accuracy) on the test dataset.")
    }
    
    // Optional: Save the model to a file
    modelFileManager.saveModel(model: neuralNetwork, toFile: modelFileURL)
    
    // Optional: Load the model from a file
    if let neuralNetworkModel = modelFileManager.loadModel(fromFile: modelFileURL) {
        // Use the loaded model for further processing
    }
}
```
## Note

Please note that the current implementation of the code has not been tested with any specific training datasets. I acknowledge that there may be potential index out of bounds errors or other issues. I have plans to test the code with new datasets and implement additional functionality in the future. Updates and tests will be conducted to improve the code's reliability and performance.

## References
1. I would like to acknowledge [Mihai Oltean](https://github.com/mihaioltean) for his ANN in C/C++, which has been a valuable reference during my studies. His code examples and insights have provided me with valuable guidance and inspiration in the field of artificial neural networks. Project can be found [here](https://github.com/mihaioltean/ANN).
2. "Deep Learning with Python" by François Chollet - This book, written by the creator of Keras, provides a practical introduction to deep learning and covers various topics, including artificial neural networks. It includes examples and code implementations using Python and Keras. Chapter 5 of the book specifically focuses on the MNIST dataset and demonstrates how to build and train neural networks for image classification tasks. You can find the book [here](https://www.manning.com/books/deep-learning-with-python-second-edition).
3. TensorFlow's MNIST tutorial - TensorFlow provides a tutorial on building a simple artificial neural network for MNIST classification using their framework. It covers the basics of data preprocessing, model construction, training, and evaluation. You can find the tutorial [here](https://www.tensorflow.org/tutorials/quickstart/beginner).
4. PyTorch's MNIST example - PyTorch also offers an example that demonstrates the implementation of a convolutional neural network (CNN) for MNIST classification. It provides a step-by-step guide on constructing the network, training it, and evaluating the results. You can find the example [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).
5. Kaggle MNIST friendly notebook can be found [here](https://www.kaggle.com/code/shantanudhakadd/mnist-beginner-friendly-notebook).


## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

