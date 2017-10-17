package com.emaraic.ArabicCharRecogntion;

import java.io.File;
import java.io.IOException;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Taha Emara Website: http://www.emaraic.com Email : taha@emaraic.com
 * Created on: Oct 12, 2017 This code generates a model for Arabic characters
 * recognition built on this paper “Arabic Handwritten Characters Recognition
 * using Convolutional Neural Network” by Ahmed El-Sawy, Mohamed Loey, and Hazem
 * EL-Bakry. 
 * The original data set from here https://github.com/mloey/Arabic-Handwritten-Characters-Dataset 
 * I used a dataset combines the data with its labels in the same file. you can find them
 * (dataset.zip) in dataset folder.
 * The output of this model gives these scores
 * ==========================Scores======================================== 
 * # of classes: 29 Accuracy: 0.9137 
 * Precision: 0.9186	(1 class excluded from average) 
 * Recall: 0.9137	(1 class excluded from average)
 * F1 Score: 0.9139	(1class excluded from average)
 * Precision, recall & F1: macro-averaged (equally weighted avg. of 29 classes)
 * ==================================================================
 */
public class ModelGenerator {

    private static final long SEED = 42;
    private static final int HEIGHT = 32;
    private static final int WIDTH = 32;
    private static final int NUM_CHANNELS = 1;
    private static final int NUM_LABELS = 29;
    private static final int BATCH_SIZE = 100;
    private static final int ITERATIONS = 1;
    private static final int LABEL_INDEX = 1024;
    private static final String PATH_TO_TRAINING_DATA = "/Users/Emaraic/Temp/ml/ahcd1/csvTrainImages 13440x1024.csv";
    private static final String PATH_TO_TESTING_DATA = "/Users/Emaraic/Temp/ml/ahcd1/csvTestImages 3360x1024.csv";

    private static final org.slf4j.Logger log = LoggerFactory.getLogger(ModelGenerator.class);

    private static DataSetIterator readCSVDataset(String csvFileClasspath, int BATCH_SIZE, int LABEL_INDEX, int numClasses)
            throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, BATCH_SIZE, LABEL_INDEX, numClasses);

        return iterator;
    }

    public static void main(String[] args) {
        try {

            DataSetIterator iterator = readCSVDataset(PATH_TO_TRAINING_DATA, BATCH_SIZE, LABEL_INDEX, NUM_LABELS);

            DataSetIterator titerator = readCSVDataset(PATH_TO_TESTING_DATA, BATCH_SIZE, LABEL_INDEX, NUM_LABELS);

            double dropOut = 0.8;
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(SEED)
                    .iterations(ITERATIONS) // Training ITERATIONS as above
                    .regularization(true)//.l2(0.0000005)
                    .learningRate(.001)
                    .weightInit(WeightInit.XAVIER)
                    .miniBatch(true)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    //updater(Updater.NESTEROVS) 
                    .updater(new Nesterovs(0.9))
                    .list()
                    .layer(0, new ConvolutionLayer.Builder(5, 5)
                            .nIn(NUM_CHANNELS)
                            .nOut(80).l2(0.0000005)
                            .activation(Activation.RELU)
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(2, new LocalResponseNormalization.Builder().build())
                    .layer(3, new ConvolutionLayer.Builder(5, 5)
                            .nOut(64)
                            .activation(Activation.RELU).l2(0.0000005)
                            .build())
                    .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(5, new LocalResponseNormalization.Builder().build())
                    .layer(6, new DenseLayer.Builder().nOut(1024).dropOut(dropOut).activation(Activation.RELU).build())
                    .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(NUM_LABELS)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .setInputType(InputType.convolutionalFlat(WIDTH, HEIGHT, 1))
                    .backprop(true).pretrain(false).build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(10));

            for (int i = 0; i < 30; i++) {
                model.fit(iterator);
            }

            log.info("Evaluate model....");
            Evaluation eval = model.evaluate(titerator);
            log.info(eval.stats(true));

            log.info("Saving model....");
            ModelSerializer.writeModel(model, new File("model.data"), true);

        } catch (IOException ex) {
            log.error(ex.getMessage());
        } catch (InterruptedException ex) {
            log.error(ex.getMessage());
        }
    }

}
