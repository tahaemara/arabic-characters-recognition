# Arabic Handwritten Characters Recognition 

In this repository, I implemented a proposed CNN in the paper "Arabic Handwritten Characters Recognition using Convolutional Neural Network" by El-Sawy, A., Loey, M., & Hazem, E. B. using Deeplearning4jlibrary.

### This repository contains:
- (ModelGenerator.java) to train the model with dataset - I provided it in dataset folder- and serialize the generated model to file (model.data). With existing network parameters, this model give a 92.29% Accuracy. You can tune these parameters to get a better accuracy. 
- The class TestModel.java is provided to test the generated model and using samples in (test_images) folder.
- I also provided a GUI application (ArabicCharactersRecognition.jar) in recogniser_executable folder to test the generated model and it gives the best three scores for the input character. 

<img src="http://emaraic.com/assets/img/posts/machine-learning/alef.png" alt="" data-canonical-src="http://emaraic.com/assets/img/posts/machine-learning/alef.png" width="400" height="400" />           <img src="http://emaraic.com/assets/img/posts/machine-learning/seen.png" alt="" data-canonical-src="http://emaraic.com/assets/img/posts/machine-learning/senn.png" width="400" height="400" />

### Note

- It would take time to train the model. For me, with 2.2 GHz Intel Core i7 on macOS, it takes nearly 1 hour (without GPU support).
- For Ubuntu (Linux) users use this commend "java -jar ArabicCharactersRecognition.jar " to run ArabicCharactersRecognition.jar from the terminal, for windows no need to do that, just double click on the jar file.

### For more info 

[http://www.emaraic.com/blog/arabic-characters-recognition](http://www.emaraic.com/blog/arabic-characters-recognition)


### Test video

https://www.youtube.com/watch?v=fvRrD4aFTu0

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/fvRrD4aFTu0/0.jpg)](https://www.youtube.com/watch?v=fvRrD4aFTu0)


### To run a maven project see this video

[https://www.youtube.com/watch?v=CDkdy3BwIqs](https://www.youtube.com/watch?v=CDkdy3BwIqs)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/CDkdy3BwIqs/0.jpg)](https://www.youtube.com/watch?v=CDkdy3BwIqs)


