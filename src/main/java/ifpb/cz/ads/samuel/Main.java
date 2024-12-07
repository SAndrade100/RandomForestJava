package ifpb.cz.ads.samuel;

import java.util.*;
import ifpb.cz.ads.samuel.tree.DecisionTree;

public class Main {
    public static void main(String[] args) {
        List<double[]> trainData = Arrays.asList(
                new double[]{2.7, 2.5},  // Classe 0
                new double[]{1.4, 2.3},  // Classe 0
                new double[]{3.3, 4.4},  // Classe 1
                new double[]{4.0, 5.0},  // Classe 2
                new double[]{6.0, 7.0}   // Classe 3
        );

        List<Integer> trainLabels = Arrays.asList(0, 0, 1, 2, 3);


        // Criar e treinar a árvore de decisão
        DecisionTree tree = new DecisionTree();
        tree.train(trainData, trainLabels);

        // Dados de teste
        List<double[]> testData = Arrays.asList(
                new double[]{1.5, 2.0},  // Próximo de classe 0
                new double[]{3.2, 3.2},  // Próximo de classe 1
                new double[]{2.0, 2.0},  // Potencialmente classe 0
                new double[]{2.7, 2.5},  // Classe 0
                new double[]{1.4, 2.3},  // Classe 0
                new double[]{3.3, 4.4},  // Classe 1
                new double[]{4.0, 5.0},  // Classe 2
                new double[]{6.0, 7.0}       );

        // Realizar previsões
        for (double[] sample : testData) {
            int prediction = tree.predict(sample);
            System.out.println("Amostra: " + Arrays.toString(sample) + " -> Previsão: " + prediction);
        }
    }
}
