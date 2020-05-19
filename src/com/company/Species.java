package com.company;
import java.util.*;

public class Species {
    public int speciesIdentifier;
    public List<NetworkWrapper> organisms;
    public NetworkWrapper representative;
    public int numChildren;

    public Species(int speciesIdentifier) {
        this.speciesIdentifier = speciesIdentifier;
        organisms = new ArrayList<>();
    }
    public void add(NetworkWrapper netWrapper) {
        organisms.add(netWrapper);
    }
    public void setRepresentative() {
        representative = organisms.get(organisms.size() / 2);
    }
}