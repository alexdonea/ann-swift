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
    var deltas:[[Double]]!
    var out:[[Double]]!
    var numIterations:Int!
    var ann_error=Double()
    var learningRate:Double!
    var epoch:Int!
    
    //constructor
    init(){
        self.numLayers=0
        self.numNeurons=nil
        self.weights=nil
        self.out=nil
        self.numIterations=1000
        self.deltas=nil
        self.epoch=0
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
            deltas[layer]=[Double(numNeurons[layer+1])]
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
    func train(trainingData:[[Double]],target:[[Double]],numData:Int){
        allocateMemory()
        initWeights()
        
        //first set the value of bias nodes to 1
        
        for layer in 0...numLayers-1{
            out[layer][numNeurons[layer]]=1;
        }
        
        for epoch in 0...numIterations{
            compute_error(trainingData: trainingData, target: target, numData: numData)
            //calling f() (File I think)
            for data_index in 0...numData{
                //forward pass
                //set input data
                for input in 0...numNeurons[0]{
                    out[0][input]=trainingData[data_index][input]
                }
                for layer in 1...numLayers{
                    for n2 in 0...numNeurons[layer]{
                        out[layer][n2]=0
                        for w in 0...(numNeurons[layer-1]+1){
                            out[layer][n2] += weights[layer-1][n2][w] * out[layer-1][w]
                        }
                        out[layer][n2]=logisticFunction(x: out[layer][n2])
                    }
                }
                
               
                //backward pass
                //update weights between last layer and hidden layer
                for n2 in 0...numNeurons[numLayers-1]{
                    deltas[numLayers-2][n2]=(out[numLayers-1][n2] - target[data_index][n2]) * out[numLayers-1][n2]*(1-out[numLayers-1][n2])
                    
                    //hidden layer
                    for neuronHiddenLayer in 0...(numNeurons[numLayers-2]+1){
                        weights[numLayers-2][n2][neuronHiddenLayer] -= learningRate * deltas[numLayers-2][n2] * out[numLayers-2][neuronHiddenLayer]
                    }
                }
                
                //weights between input layer and hidden layer
                let loop:Int=numLayers-2
                for secondLayerIndex in (numLayers-2)...loop{
                    for neuron2ndLayer in 0...numNeurons[secondLayerIndex]{
                        deltas[secondLayerIndex-1][neuron2ndLayer]=0
                        let cst:Double=out[secondLayerIndex][neuron2ndLayer] * (1 - out[secondLayerIndex][neuron2ndLayer])
                        for neuron3rdLayer in 0...numNeurons[secondLayerIndex+1]{
                            deltas[secondLayerIndex-1][neuron2ndLayer] += deltas[secondLayerIndex][neuron3rdLayer] * weights[secondLayerIndex][neuron3rdLayer][neuron2ndLayer] * cst
                        }
                        for neuron1stLayer in 0...numNeurons[secondLayerIndex-1]{
                            weights[secondLayerIndex-1][neuron2ndLayer][neuron1stLayer] -= learningRate * deltas[secondLayerIndex-1][neuron2ndLayer] * out[secondLayerIndex-1][neuron1stLayer]
                        }
                    }
                }
                
                
            }
        }
    }
    
    
    
    
    
    
    
    
    
}
