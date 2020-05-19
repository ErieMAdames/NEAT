package com.company;

import java.util.*;

public class NeuralNet {
    public HashMap<Integer, NetworkLayer> layers;
    public HashMap<Integer, Integer> layersOfNodes;
    public HashMap<Integer, Double> nodeValues;
    public HashSet<Integer> nodes;
    public int[] outputNodes;
    public int numInputs, numOutputs;
    int depth, nodeIdCounter;
    public int numLayers;
    public HashMap<Integer, int[]> innovations;

    public NeuralNet(int numInputs, int numOutputs, Random random) {
        nodeIdCounter = numInputs;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        depth = 1;
        layers = new HashMap<>();
        layersOfNodes = new HashMap<>();
        nodeValues = new HashMap<>();
        createInputLayer();
        Node[] outNodes = new Node[numOutputs];
        innovations = new HashMap<>();
        nodes = new HashSet<>();
        for (int i = 0; i < numOutputs; i++) {
            HashMap<Integer, Double> weights = new HashMap<>();
            for (int j = 0; j < numInputs; j++) {
                weights.put(j, random.nextDouble() * (2.0) - 1.0);
            }
            outNodes[i] = new Node(nodeIdCounter, weights, this);
            nodeIdCounter++;
        }
        createOutputLayer(outNodes);
        numLayers = 1;
    }
    public NeuralNet(int numInputs, int numOutputs) {
        nodeIdCounter = numInputs;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        depth = 1;
        layers = new HashMap<>();
        layersOfNodes = new HashMap<>();
        nodeValues = new HashMap<>();
        nodes = new HashSet<>();
        createInputLayer();
        createInputLayer();
    }

    public NeuralNet(NeuralNet neural) {
        nodeIdCounter = neural.nodeIdCounter;
        numInputs = neural.numInputs;
        numOutputs = neural.numOutputs;
        depth = neural.depth;
        layers = new HashMap<>();
        layersOfNodes = new HashMap<>();
        nodeValues = new HashMap<>();
        createInputLayer();
        nodes = new HashSet<>();
        numLayers = neural.numLayers;
        innovations = new HashMap<>();
        for (int inn : neural.innovations.keySet()) {
            innovate(inn, neural.innovations.get(inn)[0], neural.innovations.get(inn)[1]);
        }
        outputNodes = new int[numOutputs];
        for (NetworkLayer layer : neural.layers.values()) {
            if (layer.layerType > 0) {
                NetworkLayer newLayer = new NetworkLayer(layer.layerNumber, layer.layerType, this);
                int i = 0;
                for (Node node : layer.nodes.values()) {
                    HashMap<Integer, Double> weights = new HashMap<>(node.weights);
                    HashMap<Integer, Boolean> connectionEnabledDisabled = new HashMap<>(node.connectionEnabledDisabled);
                    Node newNode = new Node(node.id, weights, this);
                    newNode.connectionEnabledDisabled = connectionEnabledDisabled;
                    newNode.setLayer(newLayer.layerNumber);
                    newLayer.addNode(newNode);
                    nodes.add(newNode.id);
                    if (layer.layerType == 2) {
                        outputNodes[i] = newNode.id;
                        i++;
                    }
                }
                layers.put(newLayer.layerNumber, newLayer);
            }
        }
    }

    public void shiftUpLayers(int startingLayer) {
        for (int i = depth; i >= startingLayer; i--) {
            if (layers.containsKey(i)) {
                layers.get(i).layerNumber += 1;
                layers.put(i + 1, layers.get(i));
                for (Node node : layers.get(i).nodes.values()) {
                    node.setLayer(i + 1);
                }
                layers.remove(i);
            }
        }
        depth++;
    }

    public int getAndUpdateNodeIdCounter() {
        nodeIdCounter++;
        return nodeIdCounter - 1;
    }

    public void innovate(int innovationNumber, int from, int to) {
        innovations.put(innovationNumber, new int[]{from, to});
    }

    private void createInputLayer() {
        NetworkLayer inputLayer = new NetworkLayer(0, 0, this);
        inputLayer.createInputLayer();
        layers.put(0, inputLayer);
    }

    public void createLayer(Node[] nodesToAdd, int layerId) {
        NetworkLayer layer = new NetworkLayer(layerId, 1, this);
        for (Node node : nodesToAdd) {
            layer.addNode(node);
            node.setLayer(layerId);
            nodes.add(node.id);
        }
        layers.put(layerId, layer);
        numLayers++;
    }

    public void createOutputLayer(Node[] nodesToAdd) {
        NetworkLayer layer = new NetworkLayer(1, 2, this);
        for (Node node : nodesToAdd) {
            layer.addNode(node);
            node.setLayer(1);
            nodes.add(node.id);
        }
        layers.put(1, layer);
        outputNodes = new int[nodesToAdd.length];
        for (int i = 0; i < outputNodes.length; i++) {
            outputNodes[i] = nodesToAdd[i].id;
        }
    }
    public void createOutputLayer(Node[] nodesToAdd, int layerId) {
        NetworkLayer layer = new NetworkLayer(1, 2, this);
        for (Node node : nodesToAdd) {
            layer.addNode(node);
            node.setLayer(layerId);
            nodes.add(node.id);
        }
        layers.put(layerId, layer);
        outputNodes = new int[nodesToAdd.length];
        for (int i = 0; i < outputNodes.length; i++) {
            outputNodes[i] = nodesToAdd[i].id;
        }
    }

    public HashMap<Integer, Double> getOutPuts(double[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            nodeValues.put(i, inputs[i]);
        }
        HashMap<Integer, Double> outputs = new HashMap<>();
        for (int i = 0; i < depth; i++) {
            layers.get(i).layerOutput();
        }
        layers.get(depth).layerOutput();
        HashMap<Integer, Double> finalOutputs = new HashMap<>();
        Arrays.sort(outputNodes);
        for (int i = 0; i < outputNodes.length; i++) {
            finalOutputs.put(i, nodeValues.get(outputNodes[i]));
        }
        return finalOutputs;
    }

    @Override
    public String toString() {
        //NEEDS A LOT OF WORK
        StringBuilder neuralNet = new StringBuilder();
        for (int i = 1; i < layers.size(); i++) {
            HashMap<Integer, Node> nodes = layers.get(i).nodes;
            neuralNet.append(nodes.size());
            for (int node : nodes.keySet()) {
                HashMap<Integer, Double> weights = nodes.get(node).weights;
                HashMap<Integer, Boolean> endis = nodes.get(node).connectionEnabledDisabled;
                neuralNet.append("{")
                        .append(node)
                        .append(" : ");
                for (int nodeFrom : weights.keySet()) {
                    if (endis.containsKey(nodeFrom)) {
                        if (endis.get(nodeFrom)) {
                            neuralNet.append("t");
                        } else {
                            neuralNet.append("f");
                        }
                    }
                    neuralNet.append("(")
                            .append(nodeFrom)
                            .append(":")
                            .append(weights.get(nodeFrom))
                            .append(")");

                }
                neuralNet.append("}")
                        .append("--");
            }
            neuralNet.append("\n");
        }
        return neuralNet.toString();
    }
}
