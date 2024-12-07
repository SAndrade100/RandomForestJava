package ifpb.cz.ads.samuel;

import ifpb.cz.ads.samuel.tree.DecisionTree;
import org.apache.commons.csv.*;
import java.io.*;
import java.util.*;

public class SkyServerApp {
    public static void main(String[] args) {
        String csvFile = "/data/skyserver.csv";
        List<double[]> trainData = new ArrayList<>();
        List<Integer> trainLabels = new ArrayList<>();

        Map<String, Integer> classMapping = Map.of(
                "STAR", 0,
                "GALAXY", 1,
                "QUASAR", 2
        );

        try (InputStream inputStream = SkyServerApp.class.getResourceAsStream(csvFile);
             Reader reader = new InputStreamReader(Objects.requireNonNull(inputStream))) {

            Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);

            for (CSVRecord record : records) {
                double ra = Double.parseDouble(record.get("ra"));
                double dec = Double.parseDouble(record.get("dec"));
                double u = Double.parseDouble(record.get("u"));
                double g = Double.parseDouble(record.get("g"));
                double r = Double.parseDouble(record.get("r"));
                double i = Double.parseDouble(record.get("i"));
                double z = Double.parseDouble(record.get("z"));
                String starClass = record.get("class");

                trainData.add(new double[]{ra, dec, u, g, r, i, z});
                trainLabels.add(classMapping.getOrDefault(starClass, -1));
            }

        } catch (IOException | NullPointerException e) {
            e.printStackTrace();
            return;
        }

        if (trainData.isEmpty() || trainLabels.isEmpty()) {
            System.err.println("Erro: Nenhum dado carregado do CSV.");
            return;
        }

        DecisionTree tree = new DecisionTree();
        tree.train(trainData, trainLabels);

        List<double[]> testData = Arrays.asList(
                new double[]{183.5, 0.1, 19.0, 17.0, 16.0, 15.5, 15.2},
                new double[]{183.6, 0.2, 18.5, 17.5, 16.5, 16.0, 15.8},
                new double[]{183.7, 0.3, 17.5, 16.5, 15.5, 15.0, 14.8}
        );


        System.out.println("Classificando amostras:");
        for (double[] sample : testData) {
            int prediction = tree.predict(sample);
            String predictedClass = classMapping.entrySet().stream()
                    .filter(entry -> entry.getValue() == prediction)
                    .map(Map.Entry::getKey)
                    .findFirst()
                    .orElse("Unknown");
            System.out.println("Amostra: " + Arrays.toString(sample) + " -> Previsão: " + predictedClass);
        }
    }
}
