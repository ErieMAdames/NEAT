package com.company;

import java.util.*;

public class EvolutionController {
    int numInputs, numOutputs, populationSize, maxSpecies;
    double mutationRate;
    public HashMap<Integer, int[]> innovations;
    public HashMap<String, Integer> reverseInnovations;
    int innovationCounter;
    public NetworkWrapper[] networks;
    private Random random;
    private int speciesCounter;
    public List<Species> species;
    double speciationDistance;

    public EvolutionController(int numInputs, int numOutputs, int populationSize, double mutationRate) {
        this.mutationRate = mutationRate;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.populationSize = populationSize;
        this.speciationDistance = .5;
        random = new Random();
        innovations = new HashMap<>();
        reverseInnovations = new HashMap<>();
        innovationCounter = 0;
        speciesCounter = 0;
        species = new ArrayList<>();
        this.speciationDistance = .5;
        initNetworks();
    }

    public EvolutionController(int numInputs, int numOutputs, int populationSize, double mutationRate, double speciationDistance) {
        this.mutationRate = mutationRate;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.populationSize = populationSize;
        random = new Random();
        innovations = new HashMap<>();
        reverseInnovations = new HashMap<>();
        innovationCounter = 0;
        speciesCounter = 0;
        species = new ArrayList<>();
        this.speciationDistance = speciationDistance;
        initNetworks();
    }

    private void initNetworks() {
        networks = new NetworkWrapper[populationSize];
        for (int i = 0; i < numInputs; i++) {
            for (int j = numInputs; j < numOutputs + numInputs; j++) {
                innovations.put(innovationCounter, new int[]{i, j});
                innovationCounter++;
            }
        }
        int temp = innovationCounter;
        innovationCounter = 0;
        for (int i = 0; i < populationSize; i++) {
            networks[i] = new NetworkWrapper(new NeuralNet(numInputs, numOutputs, random));
            for (int j = 0; j < numInputs; j++) {
                for (int k = numInputs; k < numOutputs + numInputs; k++) {
                    networks[i].network.innovate(innovationCounter, j, k);
                    innovationCounter++;
                }
            }
            innovationCounter = 0;
        }
        innovationCounter = temp;
        speciate();
    }

    private double computeCompatabilityDistance(NetworkWrapper n1, NetworkWrapper n2) {
        List<Integer> n1List = new ArrayList<>(n1.network.innovations.keySet());
        List<Integer> n2List = new ArrayList<>(n2.network.innovations.keySet());
        n1List.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o1, o2);
            }
        });
        n2List.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o1, o2);
            }
        });
        int[] p1 = Arrays.stream(n1List.toArray()).mapToInt(o -> (int) o).toArray();
        int[] p2 = Arrays.stream(n2List.toArray()).mapToInt(o -> (int) o).toArray();
        Arrays.sort(p1);
        Arrays.sort(p2);
        double excess = 0;
        if (p1[p1.length - 1] != p2[p2.length - 1]) {
            if (p1[p1.length - 1] > p2[p2.length - 1]) {
                for (int i = p1.length - 1; i >= 0; i--) {
                    if (n2List.contains(p1[i])) {
                        break;
                    } else {
                        excess++;
                    }
                }
            } else {
                for (int i = p2.length - 1; i >= 0; i--) {
                    if (n1List.contains(p2[i])) {
                        break;
                    } else {
                        excess++;
                    }
                }
            }
        }
        double disjoint = 0 - excess;
        double totalWeightDiff = 0;
        int totalForAverage = 0;
        for (int i = 0; i < p1.length; i++) {
            if (!n2List.contains(p1[i])) {
                disjoint++;
            } else {
                int nodeTo = this.innovations.get(p1[i])[1];
                int nodeFrom = this.innovations.get(p1[i])[0];
                int layer = n1.network.layersOfNodes.get(nodeTo);
                double weight1 = n1.network.layers.get(layer).nodes.get(nodeTo).weights.get(nodeFrom);
                layer = n2.network.layersOfNodes.get(nodeTo);
                double weight2 = n2.network.layers.get(layer).nodes.get(nodeTo).weights.get(nodeFrom);
                totalWeightDiff += Math.abs(weight2 - weight1);
                totalForAverage++;
            }
        }
        for (int i = 0; i < p2.length; i++) {
            if (!n1List.contains(p2[i])) {
                disjoint++;
            }
        }
        double averageWeightDiff = totalWeightDiff / totalForAverage;
        int N = Math.max(p1.length, p2.length);
        return (disjoint / N) + (excess / N) + averageWeightDiff;
    }

    private void speciate() {
        for (int i = 0; i < this.networks.length; i++) {
            if (species.size() == 0) {
                Species newSpecies = new Species(speciesCounter);
                speciesCounter++;
                newSpecies.add(this.networks[i]);
                newSpecies.setRepresentative();
                species.add(newSpecies);
            } else {
                boolean speciesNotSet = true;
                for (Species specie : species) {
                    if (computeCompatabilityDistance(this.networks[i], specie.representative) <= speciationDistance) {
                        specie.add(this.networks[i]);
                        speciesNotSet = false;
                        break;
                    }
                }
                if (speciesNotSet) {
                    Species newSpecies = new Species(speciesCounter);
                    speciesCounter++;
                    newSpecies.add(this.networks[i]);
                    newSpecies.setRepresentative();
                    species.add(newSpecies);
                }
            }
        }
        List<Species> speciestoremove = new ArrayList<>();
        for (Species sp : species) {
            if (sp.organisms.size() > 0) {
                sp.setRepresentative();
            } else {
                speciestoremove.add(sp);
            }
        }
        for (Species sp : speciestoremove) {
            species.remove(sp);
        }
    }

    private NetworkWrapper mate(NetworkWrapper parent1, NetworkWrapper parent2) {
        double parent1Chance = parent1.getFitness() / (parent1.getFitness() + parent2.getFitness());
        if (parent1.getFitness() > parent2.getFitness()) {
            NeuralNet child = new NeuralNet(parent1.network);
            for (int innovation : parent1.network.innovations.keySet()) {
                if (parent2.network.innovations.containsKey(innovation)) {
                    if (random.nextDouble() > parent1Chance) {
                        int nodeTo = parent1.network.innovations.get(innovation)[1];
                        int nodeFrom = parent1.network.innovations.get(innovation)[0];
                        NetworkLayer childLayer = child.layers.get(child.layersOfNodes.get(nodeTo));
                        childLayer.nodes.get(nodeTo).weights.put(nodeFrom, parent2.network.layers.get(parent2.network.layersOfNodes.get(nodeTo)).nodes.get(nodeTo).weights.get(nodeFrom));
                    }
                }
            }
            NetworkWrapper childWrapper = new NetworkWrapper(child);
            mutate(childWrapper);
            return childWrapper;
        } else {
            NeuralNet child = new NeuralNet(parent2.network);
            for (int innovation : parent2.network.innovations.keySet()) {
                if (parent1.network.innovations.containsKey(innovation)) {
                    if (random.nextDouble() < parent1Chance) {
                        int nodeTo = parent1.network.innovations.get(innovation)[1];
                        int nodeFrom = parent1.network.innovations.get(innovation)[0];
                        NetworkLayer childLayer = child.layers.get(child.layersOfNodes.get(nodeTo));
                        childLayer.nodes.get(nodeTo).weights.put(nodeFrom, parent1.network.layers.get(parent1.network.layersOfNodes.get(nodeTo)).nodes.get(nodeTo).weights.get(nodeFrom));
                    }
                }
            }
            NetworkWrapper childWrapper = new NetworkWrapper(child);
            mutate(childWrapper);
            return childWrapper;
        }
    }

    private void mutate(NetworkWrapper netWrapper) {
        if (random.nextDouble() < mutationRate) {
            double p = random.nextDouble();
            if (p < .25) {
                mutateAddNode(netWrapper);
            } else if (p >= .25 && p < .5) {
                mutateAddConnection(netWrapper);
            } else if (p >= .5 && p < .75) {
                mutateRandomizeWeight(netWrapper);
            } else {
                mutateShiftWeight(netWrapper);
            }
        }
    }

    private void mutateRandomizeWeight(NetworkWrapper netWrapper) {
        int rand = random.nextInt(netWrapper.network.nodes.size());
        int ii = 0;
        for (int nodeId : netWrapper.network.nodes) {
            if (ii == rand) {
                Node node = netWrapper.network.layers.get(netWrapper.network.layersOfNodes.get(nodeId)).nodes.get(nodeId);
                rand = random.nextInt(node.weights.size());
                ii = 0;
                for (int n : node.weights.keySet()) {
                    if (rand == ii) {
                        node.weights.put(n, random.nextDouble() * (2.0) - 1.0);
                        break;
                    }
                    ii++;
                }
                break;
            }
            ii++;
        }
    }

    private void mutateShiftWeight(NetworkWrapper netWrapper) {
        int rand = random.nextInt(netWrapper.network.nodes.size());
        int ii = 0;
        for (int nodeId : netWrapper.network.nodes) {
            if (ii == rand) {
                //select random node from prev layers
                Node node = netWrapper.network.layers.get(netWrapper.network.layersOfNodes.get(nodeId)).nodes.get(nodeId);
                rand = random.nextInt(node.weights.size());
                ii = 0;
                for (int n : node.weights.keySet()) {
                    if (rand == ii) {
                        node.weights.put(n, node.weights.get(n) + random.nextDouble() * (1.0) - .5);
                        break;
                    }
                    ii++;
                }
                break;
            }
            ii++;
        }
    }

    private void mutateAddConnection(NetworkWrapper netWrapper) {
        int rand = random.nextInt(netWrapper.network.nodes.size());
        int ii = 0;
        for (int nodeId : netWrapper.network.nodes) {
            if (ii == rand) {
                //select random node from prev layers
                Node next = netWrapper.network.layers.get(netWrapper.network.layersOfNodes.get(nodeId)).nodes.get(nodeId);
                rand = random.nextInt(next.layer);
                NetworkLayer layer = netWrapper.network.layers.get(rand);
                rand = random.nextInt(layer.numNodes);
                ii = 0;
                for (int nodeFrom : layer.nodes.keySet()) {
                    if (rand == ii) {
                        if (!next.weights.containsKey(nodeFrom)) {
                            next.augmentNodeWeights(nodeFrom, random.nextDouble() * (2.0) - 1.0);
                            if (reverseInnovations.containsKey(nodeFrom + ":" + next.id)) {
                                netWrapper.network.innovate(reverseInnovations.get(nodeFrom + ":" + next.id), nodeFrom, next.id);
                            } else {
                                innovations.put(innovationCounter, new int[]{nodeFrom, next.id});
                                reverseInnovations.put(nodeFrom + ":" + next.id, innovationCounter);
                                netWrapper.network.innovate(innovationCounter, nodeFrom, next.id);
                                innovationCounter++;
                            }
                            break;
                        }
                    }
                    ii++;
                }
                break;
            }
            ii++;
        }
    }

    private void mutateAddNode(NetworkWrapper netWrapper) {
        int rand = random.nextInt(netWrapper.network.nodes.size());
        int ii = 0;
        for (int nodeId : netWrapper.network.nodes) {
            if (ii == rand) {
                Node next = netWrapper.network.layers.get(netWrapper.network.layersOfNodes.get(nodeId)).nodes.get(nodeId);
                int prevNodeId = -1;
                boolean go = true;
                while (go) {
                    ii = 0;
                    rand = random.nextInt(next.weights.size());
                    for (int node : next.weights.keySet()) {
                        if (ii == rand) {
                            prevNodeId = node;
                            break;
                        }
                        ii++;
                    }
                    if (next.connectionEnabledDisabled.get(prevNodeId)) {
                        go = false;
                    }
                }
                Node prev = netWrapper.network.layers.get(netWrapper.network.layersOfNodes.get(prevNodeId)).nodes.get(prevNodeId);
                HashMap<Integer, Double> weights = new HashMap<>();
                weights.put(prev.id, 1d);
                int midId = netWrapper.network.getAndUpdateNodeIdCounter();
                Node mid = new Node(midId, weights, netWrapper.network);
                next.disableConnection(prev.id);
                next.augmentNodeWeights(mid.id, next.weights.get(prev.id));
                netWrapper.network.shiftUpLayers(netWrapper.network.layersOfNodes.get(nodeId));
                netWrapper.network.createLayer(new Node[]{mid}, netWrapper.network.layersOfNodes.get(nodeId) - 1);
                if (reverseInnovations.containsKey(prev.id + ":" + mid.id)) {
                    netWrapper.network.innovate(reverseInnovations.get(prev.id + ":" + mid.id), prev.id, mid.id);
                } else {
                    innovations.put(innovationCounter, new int[]{prev.id, mid.id});
                    reverseInnovations.put(prev.id + ":" + mid.id, innovationCounter);
                    netWrapper.network.innovate(innovationCounter, prev.id, mid.id);
                    innovationCounter++;
                }
                if (reverseInnovations.containsKey(mid.id + ":" + next.id)) {
                    netWrapper.network.innovate(reverseInnovations.get(mid.id + ":" + next.id), mid.id, next.id);
                } else {
                    innovations.put(innovationCounter, new int[]{mid.id, next.id});
                    reverseInnovations.put(mid.id + ":" + next.id, innovationCounter);
                    netWrapper.network.innovate(innovationCounter, mid.id, next.id);
                    innovationCounter++;
                }
                break;
            }
            ii++;
        }
    }

    public void stepForward() {
        adjustFitness();
        assignNumChildren();
        killOffLowPerformers();
        normalizeFitness();
        NetworkWrapper[] newNetworks = new NetworkWrapper[populationSize];
        int c = 0;
        for (Species specie : species) {
            List<NetworkWrapper> matingPool = new ArrayList<>();
            for (NetworkWrapper organism : specie.organisms) {
                for (int i = 0; i < organism.getNormalizedFitness(); i++) {
                    matingPool.add(organism);
                }
            }
            for (int i = 0; i < specie.numChildren && c < populationSize; i++) {
                NetworkWrapper p1 = matingPool.get(random.nextInt(matingPool.size()));
                NetworkWrapper p2 = matingPool.get(random.nextInt(matingPool.size()));
                NetworkWrapper child = mate(p1, p2);
                mutate(child);
                newNetworks[c] = child;
                c++;
            }
            specie.organisms.clear();
        }
        if(c < populationSize) {
            while (c < populationSize) {
                NetworkWrapper p1 = newNetworks[random.nextInt(c)];
                NetworkWrapper p2 = newNetworks[random.nextInt(c)];
                NetworkWrapper child = mate(p1, p2);
                mutate(child);
                newNetworks[c] = child;
                c++;
            }
        }
        networks = null;
        networks = newNetworks;
        speciate();
    }

    private void adjustFitness() {
        for (Species specie : species) {
            for (NetworkWrapper organism : specie.organisms) {
                organism.setAdjustedFitness(organism.getFitness() / specie.organisms.size());
            }
        }
    }

    private void adjustFitness2() {
            for (NetworkWrapper organism : networks) {
                int shareVal = 0;
                for (NetworkWrapper organism2 : networks) {
                    shareVal += sharingFunction(computeCompatabilityDistance(organism, organism2));
                }
//                System.out.println("Shareval = " + (shareVal -1) + "tot + " + networks.length);
                organism.setAdjustedFitness(organism.getFitness() / shareVal);
        }
    }

    private void normalizeFitness() {
        for (Species specie : species) {
            double totalFitness = 0;
            for (NetworkWrapper networkWrapper : specie.organisms) {
                totalFitness += networkWrapper.getFitness();
            }
            for (NetworkWrapper networkWrapper : specie.organisms) {
                networkWrapper.setNormalizedFitness(Math.floor(populationSize * networkWrapper.getFitness() / (totalFitness * 2)));
            }
        }
    }

    private void assignNumChildren() {
        double totalAdjustedFitness = 0;
        for (int i = 0; i < networks.length; i++) {
            totalAdjustedFitness += networks[i].getAdjustedFitness();
        }
        int totalChildren = 0;
        for (Species specie : species) {
            double totalSpecieAdjustedFitness = 0;
            for (NetworkWrapper organism : specie.organisms) {
                totalSpecieAdjustedFitness += organism.getAdjustedFitness();
            }
            specie.numChildren = (int) Math.round(populationSize * totalSpecieAdjustedFitness / totalAdjustedFitness);
            totalChildren += specie.numChildren;
        }
        for (Species specie : species) {
            if (totalChildren < populationSize) {
                if (random.nextDouble() < .2) {
                    specie.numChildren++;
                    totalChildren++;
                }
            } else {
                break;
            }
        }
    }

    private void killOffLowPerformers() {
        for (Species specie : species) {
            specie.organisms.sort(new Comparator<NetworkWrapper>() {
                @Override
                public int compare(NetworkWrapper x, NetworkWrapper y) {
                    return Double.compare(x.getFitness(), y.getFitness());
                }
            });
            int numToKill = specie.organisms.size() / 2;
            for (int i = 0; i < numToKill; i++) {
                specie.organisms.remove(0);
            }
            if (specie.organisms.size() > 0) {
                specie.setRepresentative();

            } else {
                species.remove(specie);
            }
        }
    }

    private double sharingFunction(double dist) {
        if (dist > speciationDistance) {
            return 0;
        } else {
            return 1;
        }
    }
}
