package com.company;

import com.sun.javafx.image.impl.IntArgb;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Main {

    public static double[][] inputs = new double[][]{
            new double[]{1, 2, 3},
            new double[]{2, 3, 4},
            new double[]{4, 5, 6},
            new double[]{8, 9, 10}
    };

    public static void main(String[] args) {
        Random random = new Random();
        // write your code here
        EvolutionController controller = new EvolutionController(3, 2, 150, .1);
        for (int i = 0; i < 50; i++) {
            if (i % 500 == 0) {
                System.out.println("step " + i);
            }
            for (NetworkWrapper wrapper : controller.networks) {
                List<HashMap<Integer, Double>> outputs = new ArrayList<>();
                for (double[] input : inputs) {
                    outputs.add(wrapper.network.getOutPuts(input));
                }
                wrapper.setFitness(fitnessFunction(outputs));
            }
            controller.stepForward();
        }
        int s = 1;
        double maxFit = Double.MIN_VALUE;
        Species max = new Species(0);
        for (Species specie : controller.species) {
            for (NetworkWrapper wrapper : controller.networks) {
                List<HashMap<Integer, Double>> outputs = new ArrayList<>();
                for (double[] input : inputs) {
                    outputs.add(wrapper.network.getOutPuts(input));
                }
                wrapper.setFitness(fitnessFunction(outputs));
            }
            if (specie.representative.getFitness() > maxFit) {
                maxFit = specie.representative.getFitness();
                max = specie;
            }
            s++;
        }
        for (int node : max.representative.network.nodes) {
            System.out.print(node + " : ");
        }
        System.out.println();
        for (double[] input : inputs) {
            HashMap<Integer, Double> outputs = max.representative.network.getOutPuts(input);
            System.out.println("is : " + outputs.get(0) + " : " + outputs.get(1));
        }
        try {
            neuralNetToFile(max.representative.network, "network.txt");
            NeuralNet n = fileToNeuralNet("network.txt");
            for (double[] input : inputs) {
                HashMap<Integer, Double> outputs = n.getOutPuts(input);
                System.out.println("is : " + outputs.get(0) + " : " + outputs.get(1));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static double fitnessFunction(List<HashMap<Integer, Double>> outputs) {
        double out = 0;
        for (HashMap<Integer, Double> output : outputs) {
            out += 2 - Math.abs(1 - output.get(0));
            out += 2 - Math.abs(1 + output.get(1));
        }
        return out;
    }

    public static void neuralNetToFile(NeuralNet neuralNet, String fileName) throws IOException {
        FileWriter net = new FileWriter(fileName);
        net.write(neuralNet.numInputs + "\n");
        net.write(neuralNet.numOutputs + "\n");
        net.write(neuralNet.depth + "\n");
        net.write(neuralNet.toString());
        net.close();
    }

    public static NeuralNet fileToNeuralNet(String fileName) throws IOException {
        File netFile = new File(fileName);
        Scanner scanner = new Scanner(netFile);
        int numLayers = 0;
        int numInputs = Integer.parseInt(scanner.nextLine());
        int numOutputs = Integer.parseInt(scanner.nextLine());
        int depth = Integer.parseInt(scanner.nextLine());
        NeuralNet neuralNet = new NeuralNet(numInputs, numOutputs);
        while (scanner.hasNextLine()) {
            numLayers++;
            String data = scanner.nextLine();
            int fstBracket = data.indexOf('{');
            int numNodes = Integer.parseInt(data.substring(0, fstBracket));
            Node[] nodes = new Node[numNodes];
            int cNode = 0;
            int index = fstBracket;
            while (index < data.length()) {
                if (data.charAt(index) == '{') {
                    int space = data.indexOf(' ', index);
                    int end = data.indexOf('}', index);
                    int nodeId = Integer.parseInt(data.substring(index + 1, space));
                    HashMap<Integer, Double> weights = new HashMap<>();
                    HashMap<Integer, Boolean> connectionEnabledDisabled = new HashMap<>();
                    int i = data.indexOf('(', index + 1) - 1;
                    while (i < end) {
                        char tf = data.charAt(i);
                        boolean bool = tf == 't';
                        i += 2;
                        int colon = data.indexOf(':', i);
                        int nodeIdWeight = Integer.parseInt(data.substring(i, colon));
                        connectionEnabledDisabled.put(nodeIdWeight, bool);
                        i = colon + 1;
                        int p = data.indexOf(')', i);
                        double weight = Double.parseDouble(data.substring(i, p));
                        weights.put(nodeIdWeight, weight);
                        i = p + 1;
                    }
                    Node node = new Node(nodeId, weights, neuralNet);
                    nodes[cNode] = node;
                    cNode++;
                    node.connectionEnabledDisabled = connectionEnabledDisabled;
                    index = end;
                } else {
                    index++;
                }
            }
            if (numLayers == depth) {
                neuralNet.createOutputLayer(nodes, numLayers);
            } else {
                neuralNet.createLayer(nodes, numLayers);
            }
        }
        neuralNet.numLayers = numLayers;
        neuralNet.depth = numLayers;
//        System.out.println(neuralNet.toString());
        scanner.close();
        return neuralNet;
    }
}
