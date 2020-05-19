package com.company;

public class NetworkWrapper {
    public NeuralNet network;
    private double fitness;
    private double adjustedFitness;
    private double normalizedFitness;
    public NetworkWrapper(NeuralNet neuralNet) {
        network = neuralNet;
    }
    public void setFitness(double fitness) {
        this.fitness = fitness;
    }
    public double getFitness() {
        return fitness;
    }
    public void setAdjustedFitness(double adjustedFitness) {
        this.adjustedFitness = adjustedFitness;
    }
    public double getAdjustedFitness() {
        return adjustedFitness;
    }
    public void setNormalizedFitness(double normalizedFitness) {
        this.normalizedFitness = normalizedFitness;
    }
    public double getNormalizedFitness() {
        return normalizedFitness;
    }
    @Override
    public String  toString() {
        return network.toString();
    }
}
