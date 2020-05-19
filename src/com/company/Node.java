package com.company;

import java.util.*;

public class Node {
    public int id;
    public int layer;
    public HashMap<Integer, Double> weights;
    public HashMap<Integer, Boolean> connectionEnabledDisabled;
    public boolean isInput;
    public NeuralNet parent;

    public Node(int id, NeuralNet parent) {
        layer = 0;
        isInput = true;
        this.id = id;
        this.parent = parent;
        connectionEnabledDisabled = new HashMap<>();
        parent.layersOfNodes.put(id, layer);
    }

    public void setLayer(int layer) {
        this.layer = layer;
        parent.layersOfNodes.put(id, layer);
    }

    public Node(int id, HashMap<Integer, Double> weights, NeuralNet parent) {
        isInput = false;
        this.id = id;
        this.weights = weights;
        this.parent = parent;
        connectionEnabledDisabled = new HashMap<>();
        for (int node : weights.keySet()) {
            connectionEnabledDisabled.put(node, true);
        }
    }

    public void augmentNodeWeights(int nodeFrom, double weight) {
        weights.put(nodeFrom, weight);
        connectionEnabledDisabled.put(nodeFrom, true);
    }

    public void disableConnection(int nodeFromId) {
        connectionEnabledDisabled.put(nodeFromId, false);
    }

    public double getOutput() {
        double acc = 0;
        if (layer == 0) {
            return parent.nodeValues.get(id);
        } else {
            for (int key : connectionEnabledDisabled.keySet()) {
                if (connectionEnabledDisabled.get(key)) {
                    acc += parent.nodeValues.get(key) * weights.get(key);
                }
            }
        }
        return Math.tanh(acc);
    }
}
