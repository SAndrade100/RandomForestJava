package ifpb.cz.ads.samuel.tree;

import java.util.*;

public class DecisionTree {
    private Node root;

    // Classe para representar os nós da árvore
    private static class Node {
        double splitValue;
        int featureIndex;
        Node left;
        Node right;
        Integer label; // Rótulo do nó (se for folha)

        public Node(Integer label) {
            this.label = label;
        }

        public Node(double splitValue, int featureIndex) {
            this.splitValue = splitValue;
            this.featureIndex = featureIndex;
        }
    }

    // Método público para treinar a árvore
    public void train(List<double[]> data, List<Integer> labels) {
        root = buildTree(data, labels);
    }

    // Método recursivo para construir a árvore
    private Node buildTree(List<double[]> data, List<Integer> labels) {
        // Caso base: se todos os rótulos forem iguais, retorne uma folha
        if (labels.stream().distinct().count() == 1) {
            return new Node(labels.get(0));
        }

        // Encontre o melhor ponto de divisão
        int bestFeature = -1;
        double bestSplitValue = 0;
        double bestGini = Double.MAX_VALUE;

        for (int feature = 0; feature < data.get(0).length; feature++) {
            for (double[] sample : data) {
                double splitValue = sample[feature];
                double gini = calculateGini(data, labels, feature, splitValue);

                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = feature;
                    bestSplitValue = splitValue;
                }
            }
        }

        // Divida os dados com base na melhor divisão encontrada
        List<double[]> leftData = new ArrayList<>();
        List<double[]> rightData = new ArrayList<>();
        List<Integer> leftLabels = new ArrayList<>();
        List<Integer> rightLabels = new ArrayList<>();

        for (int i = 0; i < data.size(); i++) {
            if (data.get(i)[bestFeature] <= bestSplitValue) {
                leftData.add(data.get(i));
                leftLabels.add(labels.get(i));
            } else {
                rightData.add(data.get(i));
                rightLabels.add(labels.get(i));
            }
        }

        // Crie o nó atual e chame recursivamente para os filhos
        Node node = new Node(bestSplitValue, bestFeature);
        node.left = buildTree(leftData, leftLabels);
        node.right = buildTree(rightData, rightLabels);
        return node;
    }

    private double calculateGini(List<double[]> data, List<Integer> labels, int feature, double splitValue) {
        int leftCount = 0, rightCount = 0;
        Map<Integer, Integer> leftLabelCount = new HashMap<>();
        Map<Integer, Integer> rightLabelCount = new HashMap<>();

        for (int i = 0; i < data.size(); i++) {
            if (data.get(i)[feature] <= splitValue) {
                leftCount++;
                leftLabelCount.put(labels.get(i), leftLabelCount.getOrDefault(labels.get(i), 0) + 1);
            } else {
                rightCount++;
                rightLabelCount.put(labels.get(i), rightLabelCount.getOrDefault(labels.get(i), 0) + 1);
            }
        }

        double leftGini = 1.0, rightGini = 1.0;

        for (int count : leftLabelCount.values()) {
            double proportion = (double) count / leftCount;
            leftGini -= proportion * proportion;
        }

        for (int count : rightLabelCount.values()) {
            double proportion = (double) count / rightCount;
            rightGini -= proportion * proportion;
        }

        return ((double) leftCount / data.size()) * leftGini +
                ((double) rightCount / data.size()) * rightGini;
    }

    // Método público para realizar predições
    public int predict(double[] features) {
        Node node = root;
        while (node.label == null) {
            if (features[node.featureIndex] <= node.splitValue) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return node.label;
    }
}
