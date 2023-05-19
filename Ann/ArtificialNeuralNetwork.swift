//
//  ArtificialNeuralNetwork.swift
//  Ann
//
//  Created by Alexandru Donea on 19/05/2023.
//  Copyright © 2023 Alexandru Donea. All rights reserved.
//

import Foundation

class ArtificialNeuralNetwork: Encodable, Decodable {
    // The number of nodes in the input layer
    var inputNodes: Int

    // The number of nodes in the hidden layer
    var hiddenNodes: Int

    // The number of nodes in the output layer
    var outputNodes: Int

    // The learning rate used in the training process
    var learningRate: Double

    // The weights connecting the input layer to the hidden layer
    var weightsIH: [[Double]]

    // The weights connecting the hidden layer to the output layer
    var weightsHO: [[Double]]

    // The number of training epochs
    var epoch: Int

    // The current error or loss value
    var error: Double

    // The number of incorrectly classified samples during training or testing
    var numIncorrectClassified: Int

    
    // Initializes a neural network with the specified number of nodes in each layer and learning rate
    init(inputNodes: Int, hiddenNodes: Int, outputNodes: Int, learningRate: Double) {
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate
        
        // Initialize the weights connecting the input layer to the hidden layer
        self.weightsIH = [[Double]](repeating: [Double](repeating: 0, count: hiddenNodes), count: inputNodes)
        
        // Initialize the weights connecting the hidden layer to the output layer
        self.weightsHO = [[Double]](repeating: [Double](repeating: 0, count: outputNodes), count: hiddenNodes)
        
        // Calculate the range for random initialization of weights using Xavier initialization
        let range = sqrt(6.0 / Double(inputNodes + hiddenNodes))
        
        // Initialize the weightsIH matrix with random values in the range [-range, range]
        for i in 0..<inputNodes {
            for j in 0..<hiddenNodes {
                self.weightsIH[i][j] = Double.random(in: -range...range)
            }
        }
        
        // Initialize the weightsHO matrix with random values in the range [-range, range]
        for i in 0..<hiddenNodes {
            for j in 0..<outputNodes {
                self.weightsHO[i][j] = Double.random(in: -range...range)
            }
        }
        
        self.epoch = 0
        self.error = 0.0
        self.numIncorrectClassified = 0
    }
    
    // Applies the Rectified Linear Unit (ReLU) activation function to the input value
    // by returning the input if it is positive, or zero otherwise
    func relu(_ x: Double) -> Double {
        return max(0, x)
    }
    
    // Applies the Softmax activation function to the input array of values
    // by transforming them into probabilities that sum up to 1
    func softmax(_ x: [Double]) -> [Double] {
        // Find the maximum value in the input array
        let maxVal = x.max() ?? 0.0
        
        // Compute the exponential values of each element in the input array
        // after subtracting the maximum value to prevent numerical instability
        let expValues = x.map { exp($0 - maxVal) }
        
        // Compute the sum of the exponential values
        let sumExpValues = expValues.reduce(0, +)
        
        // Normalize the exponential values by dividing each value by the sum
        // to obtain probabilities that sum up to 1
        return expValues.map { $0 / sumExpValues }
    }
    
    // Performs feedforward propagation by computing the outputs of the neural network
    // given the input values
    func feedForward(_ inputs: [Double]) -> [Double] {
        // Create arrays to store the values of the hidden layer and output layer
        var hiddenLayer: [Double] = [Double](repeating: 0, count: hiddenNodes)
        var outputLayer: [Double] = [Double](repeating: 0, count: outputNodes)
        
        // Compute the values of the hidden layer
        for j in 0..<hiddenNodes {
            var sum = 0.0
            for i in 0..<inputNodes {
                sum += inputs[i] * weightsIH[i][j]
            }
            // Apply the ReLU activation function to the sum and store the result
            hiddenLayer[j] = relu(sum)
        }
        
        // Compute the values of the output layer
        for j in 0..<outputNodes {
            var sum = 0.0
            for i in 0..<hiddenNodes {
                sum += hiddenLayer[i] * weightsHO[i][j]
            }
            // Store the sum in the output layer
            outputLayer[j] = sum
        }
        
        // Apply the Softmax activation function to the output layer and return the result
        return softmax(outputLayer)
    }
    
    // Training the neural network
    func train(_ inputs: [[Double]], _ targets: [[Double]], batchSize: Int, epochs: Int,
               learningRateDecay: Double, dropoutRate: Double) {
        let numSamples = inputs.count
        let numBatches = Int(ceil(Double(numSamples) / Double(batchSize)))
        var totalError = 0.0
        var numIncorrectClassified = 0
        
        for epoch in 1...epochs {
            
            // Shuffle the training data
            let shuffledData = zip(inputs, targets).shuffled()
            let shuffledInputs = shuffledData.map { $0.0 }
            let shuffledTargets = shuffledData.map { $0.1 }
            
            // Learning rate decay
            let adjustedLearningRate = learningRate / (1.0 + Double(epoch) * learningRateDecay)
            
            // Mini-batch training
            for batch in 0..<numBatches {
                let startIndex = batch * batchSize
                let endIndex = min(startIndex + batchSize, numSamples)
                let batchInputs = Array(shuffledInputs[startIndex..<endIndex])
                let batchTargets = Array(shuffledTargets[startIndex..<endIndex])
                
                
                
                // Initialize weight updates
                var weightIHUpdates = [[Double]](repeating: [Double](repeating: 0, count: hiddenNodes), count: inputNodes)
                var weightHOUpdates = [[Double]](repeating: [Double](repeating: 0, count: outputNodes), count: hiddenNodes)
                
                for (inputs, targets) in zip(batchInputs, batchTargets) {
                    // Feedforward
                    var hiddenLayer: [Double] = [Double](repeating: 0, count: hiddenNodes)
                    var outputLayer: [Double] = [Double](repeating: 0, count: outputNodes)
                    
                    for j in 0..<hiddenNodes {
                        var sum = 0.0
                        for i in 0..<inputNodes {
                            sum += inputs[i] * weightsIH[i][j]
                        }
                        hiddenLayer[j] = relu(sum)
                        
                        // Dropout regularization
                        if dropoutRate > 0.0 {
                            let mask = Double.random(in: 0...1) > dropoutRate ? 1.0 : 0.0
                            hiddenLayer[j] *= mask
                        }
                    }
                    
                    for j in 0..<outputNodes {
                        var sum = 0.0
                        for i in 0..<hiddenNodes {
                            sum += hiddenLayer[i] * weightsHO[i][j]
                        }
                        outputLayer[j] = sum
                    }
                    
                    let predictedTargets = softmax(outputLayer)
                    
                    // Backpropagation
                    var outputErrors: [Double] = [Double](repeating: 0, count: outputNodes)
                    var hiddenErrors: [Double] = [Double](repeating: 0, count: hiddenNodes)
                    
                    for i in 0..<outputNodes {
                        outputErrors[i] = targets[i] - predictedTargets[i]
                    }
                    
                    for i in 0..<hiddenNodes {
                        var error = 0.0
                        for j in 0..<outputNodes {
                            error += outputErrors[j] * weightsHO[i][j]
                        }
                        hiddenErrors[i] = error
                    }
                    
                    // Update weight updates
                    for i in 0..<inputNodes {
                        for j in 0..<hiddenNodes {
                            weightIHUpdates[i][j] += hiddenErrors[j] * (hiddenLayer[j] > 0.0 ? inputs[i] : 0.0)
                        }
                    }
                    
                    for i in 0..<hiddenNodes {
                        for j in 0..<outputNodes {
                            weightHOUpdates[i][j] += outputErrors[j] * hiddenLayer[i]
                        }
                    }
                    
                    // Calculate error and number of incorrectly classified samples
                    totalError += -log(predictedTargets[Int(targets[0])])
                    if predictedTargets.argmax() != Int(targets[0]) {
                        numIncorrectClassified += 1
                    }
                }
                
                // Update the weights
                for i in 0..<inputNodes {
                    for j in 0..<hiddenNodes {
                        weightsIH[i][j] += adjustedLearningRate * weightIHUpdates[i][j] / Double(batchSize)
                    }
                }
                
                for i in 0..<hiddenNodes {
                    for j in 0..<outputNodes {
                        weightsHO[i][j] += adjustedLearningRate * weightHOUpdates[i][j] / Double(batchSize)
                    }
                }
            }
            
            // Calculate average error and number of incorrectly classified samples
            let averageError = totalError / Double(numSamples)
            let accuracy = 1.0 - Double(numIncorrectClassified) / Double(numSamples)
            
            // Print epoch information
            print("Epoch \(epoch): Average Error = \(averageError), Accuracy = \(accuracy)")
        }
        
        self.epoch += epochs
        self.error = totalError / Double(numSamples)
        self.numIncorrectClassified = numIncorrectClassified
    }
    
    // Validate the neural network on a validation dataset
    func validate(inputs: [[Double]], targets: [[Double]]) -> (Double, Double)? {
        guard inputs.count == targets.count else {
            print("Mismatched number of inputs and targets.")
            return nil
        }
        
        var totalError = 0.0
        var numIncorrectClassified = 0
        
        for (input, target) in zip(inputs, targets) {
            let predicted = predict(input)
            guard let trueClassIndex = target.argmax(),
                  let trueClass = Int(exactly: trueClassIndex) else {
                print("Prediction failed for input: \(input)")
                return nil
            }
            
            let predictedClass = predicted.argmax()
            
            if predictedClass != trueClass {
                numIncorrectClassified += 1
            }
            
            totalError += -log(predicted[trueClass])
        }
        
        let averageError = totalError / Double(inputs.count)
        let accuracy = 1.0 - Double(numIncorrectClassified) / Double(inputs.count)
        
        return (averageError, accuracy)
    }
    
    // Make a prediction for an individual input
       func predict(_ input: [Double]) -> [Double] {
           return feedForward(input)
       }
}


// Return index of max value in Array Element
extension Array where Element == Double {
    func argmax() -> Int? {
        guard let maxElement = self.max() else {
            return nil
        }
        return firstIndex(of: maxElement)
    }
}



