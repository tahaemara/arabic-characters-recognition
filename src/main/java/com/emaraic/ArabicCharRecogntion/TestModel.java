package com.emaraic.ArabicCharRecogntion;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Taha Emara 
 * Website: http://www.emaraic.com 
 * Email : taha@emaraic.com
 * Created on: Oct 16, 2017
 * 
 * You can use this demo to test the generated model from example ModelGenerator.java
 * You can use the images files in test_images folder. 
 * This program accepts input image such that char color is black with black background with any size as  images in test_images folder,
 * you can do some image processing to make it accepts any image color.  
 */
public class TestModel {

    private static final String[] ALPHABET = {"alef-ألف", "beh-باء", "teh-تاء", "theh-ثاء", "jeem-جيم", "hah-حاء", "khah-خاء", "dal-دال", "thal-ذال",
        "reh-راء", "zah-زاى", "seen-سين", "sheen-شين", "sad-صاد", "dad-ضاد", "tah-طاء", "zah-ظاء", "ain-عين",
        "ghain-غين", "feh-فاء", "qaf-قاف", "kaf-كاف", "lam-لام", "meem-ميم", "noon-نون", "heh-هاء", "waw-واو", "yeh-ياء"};
    
    private static final String PATH_TO_MODEL_FILE="model.data";


    private static double[][] imageToMat(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        double[][] imgArr = new double[width][height];
        Raster raster = img.getData();
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                imgArr[i][j] = raster.getSample(i, j, 0);
            }
        }
        return imgArr;
    }

    public static BufferedImage convertToBufferedImage(Image img) {
        if (img instanceof BufferedImage) {
            return (BufferedImage) img;
        }

        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }

  

    public static void main(String[] args) {
        MultiLayerNetwork restored;
        try {
            File locationToSave = new File(PATH_TO_MODEL_FILE);
            restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
            BufferedImage img = ImageIO.read(new File("/Users/Emaraic/Desktop/UNKNOWN1.png"));
            BufferedImage image = convertToBufferedImage(img.getScaledInstance(32, 32, Image.SCALE_DEFAULT)); //crop image to size 32 X 32
            double[][] values = imageToMat(image);//convert image data to 2d matrix
            
            double[] flat = Arrays.stream(values)// convert 2d array to 1d array
                    .flatMapToDouble(Arrays::stream)
                    .toArray();
            double [] temp=new double [flat.length];
            for (int i = 0; i < flat.length; i++) { //convert (Black char and weight background) image to (weight char and black background) image
                if(flat[i]==255)
                    temp[i]=0.0;
               else
                temp[i]= 255.0; 
            }
            
            INDArray ss = Nd4j.create(temp, new int[]{32, 32}); // reshape mat to 32X32 just fot printing it
            System.out.println(ss);
          
            INDArray ss1 = ss.reshape(new int[]{1, 1024}); // flat it again

            INDArray output = restored.output(ss1); //feed input vector to restored nn and receive the output vector 
            System.out.println(output);
            
            int idx = Nd4j.getExecutioner().execAndReturn(new IAMax(output)).getFinalResult();//get index of best result
            
            System.out.println("Result is : "+ALPHABET[idx - 1]);
            
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
