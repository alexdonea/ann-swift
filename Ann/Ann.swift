//
//  Ann.swift
//  Ann
//
//  Created by Alexandru Cristian Donea on 22/05/2018.
//  Copyright © 2018 University Projects. All rights reserved.
//

import Foundation

class Ann{
    //initialize variables
    var numLayers:Int!
    var numNeurons:[Int]!
    var weights:[[[Double]]]!
    var deltas:[Double]!
    var out:[[Double]]!
    var numIterations:Int!
    var ann_error=Double()
    var learningRate:Double!
    var epoch:Int!
    
    //constructor
    init(){
        numLayers=0
        numNeurons=nil
        weights=nil
        out=nil
        numIterations=1000
        deltas=nil
        epoch=0
    }
    //destructor
    deinit {
        //delete all values from arrays
        releaseMemory()
    }
    //setters for variables
    func setNumLayers(numLayers_:Int){
        self.numLayers=numLayers_
        if(numLayers_ > 0){
            numNeurons.reserveCapacity(numLayers_)
        }
    }
    func setNumNeurons(layerIndex:Int,numNeurons:Int){
        self.numNeurons[layerIndex] = numNeurons
    }
    func setNumIterations(numIterations_:Int){
        self.numIterations=numIterations_
    }
    func setLearningRate(setLearningRate_:Double){
        self.learningRate=setLearningRate_
    }
    //getters for variables
    func getNumLayers()->Int{
        return self.numLayers
    }
    func getNumNeurons(layerIndex:Int)->Int{
        return self.numNeurons[layerIndex]
    }
    func getWeight(layerIndex:Int, neuronIndex:Int,weightIndex:Int)->Double{
        return 0.0; //must be implemented
    }
    func getNumIterations()->Int{
        return self.numIterations
    }
    func allocateMemory(){
        //allocate weights
        out.reserveCapacity(numLayers)
        for i in 0...numLayers-1{
            out[i] = [Double(numNeurons[i+1])]
            out[numLayers-1] = [Double(numNeurons[numLayers-1])]
        }
        //allocae weights
        weights.reserveCapacity(numLayers-1)
        for layer in 0...numLayers-1{
            weights[layer]=[[Double(numNeurons[layer+1])]]
            for neuron in 0...numNeurons[layer+1]{
                weights[layer][neuron]=[Double(numNeurons[layer+1])]
            }
        }
        //allocate deltas
        deltas.reserveCapacity(numLayers-1)
        for layer in 0...numLayers-1{
            deltas[layer]=Double(numNeurons[layer+1])
        }
    }
    func releaseMemory(){
        out.removeAll()
        weights.removeAll()
        deltas.removeAll()
        numNeurons.removeAll()
        numNeurons=nil
    }
  
    func getError()->Double{
        return ann_error;
    }
    func logisticFunction(x:Double)->Double{
        return 1/(1+exp(-x))
    }
    func setLearningRate(newLearningRate:Double){
        self.learningRate=newLearningRate
    }
    func compute_error(trainingData:[[Double]],target:[[Double]],numData:Int){
        ann_error=0
        for dataIndex in 0...numData{
            //set input data
            for input in 0...numNeurons[0]{
                out[0][input]=trainingData[dataIndex][input]
            }
            //compute out for each other layer
            for layer in 1...numLayers{
                for n2 in 0...numNeurons[layer]{
                    out[layer][n2]=0;
                    for w in 0...numNeurons[layer-1]+1{
                      out[layer][n2]+=weights[layer-1][n2][w]*out[layer-1][w]
                      out[layer][n2]=logisticFunction(x: out[layer][n2])
                    }
                }
            }
            for neuron in 0...numNeurons[numLayers-1]{
                ann_error += 1 / 2.0 * (out[numLayers-1][neuron] - target[dataIndex][neuron] * (out[numLayers-1][neuron]-target[dataIndex][neuron]))
            }
        }
        
    }
    func initWeights(){
        for layer in 0...numLayers-1{
            for neuron in 0...numNeurons[layer+1]{
                for w in 0...numNeurons[layer]+1{
                    //Unfortunadely I didn't found the function that will return maxim of random generated number
                    //I will implement or create it later
                    weights[layer][neuron][w]=Double(arc4random())/0.5
                }
            }
        }
    }
    
    
    func getEpoch()->Int{
        return self.epoch
    }
    
    //declarig typedef (swift typelias)
    
    //training data
    func train(trainingData:[[Double]],target:[[Double]],numData:Int, f){
        
    }
    
    
    
    
    
    
    
    
    
}
