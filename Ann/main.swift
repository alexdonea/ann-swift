//
//  main.swift
//  Ann
//
//  Created by Alexandru Donea on 19/05/2023.
//  Copyright © 2023 Alexandru Donea. All rights reserved.
//

import Foundation

// Create an instance of the model file manager
let modelFileManager = ModelFileManager()

// Path to the 'mnist' folder inside the project
let path = "/your-path/ann-swift"
let trainDataFile = "\(path)/mnist/mnist_train.txt"
let testDataFile = "\(path)/mnist/mnist_test.txt"

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
} else {
    print("Dataset files cannot be found.")
}
