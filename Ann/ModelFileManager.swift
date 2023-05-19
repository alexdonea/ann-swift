//
//  ModelFileManager.swift
//  Ann
//
//  Created by Alexandru Donea on 19/05/2023.
//  Copyright © 2023 Alexandru Donea. All rights reserved.
//

import Foundation

class ModelFileManager {
    // Allocates memory for training data and target arrays
    func allocateTrainingData(trainingData: inout [[Double]], target: inout [[Double]], numTrainingData: Int, numVariables: Int, numOutputs: Int) {
        target = Array(repeating: Array(repeating: 0, count: numOutputs), count: numTrainingData)
        trainingData = Array(repeating: Array(repeating: 0, count: numVariables), count: numTrainingData)
    }
    
    // Reads the data from a file and returns the training data, test data, target, and other information
    func readFile(filename: String) -> ([[Double]], [[Double]], Int, Int, Int)? {
        print("Load file:", filename)
        let fileURL = URL(fileURLWithPath: filename)
        
        // Read file content
        guard let fileContent = try? String(contentsOf: fileURL, encoding: .utf8) else {
            return nil
        }
        
        var lines = fileContent.components(separatedBy: .newlines)
        
        // Extract header values
        guard let headerLine = lines.first else {
            print("Insufficient data in the file.")
            return nil
        }
        
        let headerValues = headerLine.components(separatedBy: .whitespaces)
        guard headerValues.count == 3,
              let numTrainingData = Int(headerValues[0]),
              let numVariables = Int(headerValues[1]),
              let numOutputs = Int(headerValues[2]) else {
            print("Invalid header values in the file: \(headerValues)")
            return nil
        }
        
        // Remove header line and empty lines
        lines.removeFirst()
        lines = lines.filter({ !$0.isEmpty })
        
        var trainingData: [[Double]] = []
        var target: [[Double]] = []
        
        allocateTrainingData(trainingData: &trainingData, target: &target, numTrainingData: numTrainingData, numVariables: numVariables, numOutputs: numOutputs)
        
        // Parse the data lines
        for i in 0..<numTrainingData {
            let lineValues = lines[i].components(separatedBy: .whitespaces).compactMap { Double($0) }
            
            guard lineValues.count == numVariables + 1 else {
                print("Invalid number of values on line \(i)")
                return nil
            }
            
            for j in 0..<numVariables {
                trainingData[i][j] = lineValues[j]
            }
            
            let classIndex = Int(lineValues[numVariables])
            target[i][classIndex] = 1
        }
        
        return (trainingData, target, numTrainingData, numVariables, numOutputs)
    }
    
    // Load training and test dataset
    func loadTrainAndTestData(trainLocation: String, testLocation: String) -> (Bool, [[Double]], [[Double]], [[Double]], [[Double]], Int, Int) {
        if let (trainingData, trainTarget, _, numTrainVariables, numTrainOutputs) = readFile(filename: trainDataFile) {
            
            if let (testData, testTarget, _, _, _) = readFile(filename: testLocation) {
                return (true, trainingData, trainTarget, testData, testTarget, numTrainVariables, numTrainOutputs)
            }
            return (false, [[]], [[]], [[]], [[]], 0, 0)
        }
        return (false, [[]], [[]], [[]], [[]], 0, 0)
    }
    
    // Saves the NeuralNetwork model to a file
    func saveModel(model: ArtificialNeuralNetwork, toFile fileURL: URL) {
        do {
            let encoder = JSONEncoder()
            let modelData = try encoder.encode(model)
            try modelData.write(to: fileURL)
            print("Model saved successfully.")
        } catch {
            print("Failed to save model:", error)
        }
    }
    
    // Loads the NeuralNetwork model from a file
    func loadModel(fromFile fileURL: URL) -> ArtificialNeuralNetwork? {
        do {
            let modelData = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            let model = try decoder.decode(ArtificialNeuralNetwork.self, from: modelData)
            print("Model loaded successfully.")
            return model
        } catch {
            print("Failed to load model:", error)
            return nil
        }
    }
}
