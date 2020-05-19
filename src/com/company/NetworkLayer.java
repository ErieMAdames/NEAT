package com.company;

import java.util.HashMap;

public class NetworkLayer {
    public int layerType;
    public int layerNumber;
    public HashMap<Integer, Node> nodes;
    public int numNodes;
    public NeuralNet parent;

    public NetworkLayer(int layerNumber, int layerType, NeuralNet parent) {
        this.layerNumber = layerNumber;
        this.layerType = layerType;
        this.parent = parent;
        nodes = new HashMap<>();
        numNodes = 0;
    }
    public void createInputLayer() {
        if (layerType == 0) {
            for (int i = 0; i < parent.numInputs; i++) {
                nodes.put(i, new Node(i, parent));
                numNodes++;
            }
        }
    }
    public void addNode(Node node) {
        if (layerType > 0) {
            node.setLayer(layerNumber);
            nodes.put(node.id, node);
            numNodes++;
        }
    }
    public void layerOutput() {
        if (layerType != 0) {
            for (int n : nodes.keySet()) {
                parent.nodeValues.put(n,nodes.get(n).getOutput());
            }
        }
    }
}