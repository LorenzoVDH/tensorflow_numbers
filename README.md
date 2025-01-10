My first TensorFlow project.

I want to immerse myself in Machine Learning (and AI in general), so what else than to start with the classic mnist numbers dataset? 
This dataset contains the numbers 0 to 9 (labeled). 
It gets prepared, by deviding each value of the greyscale imageset by 255, fitting it between 0-1.
Then the arrays get reshaped (28 x 28, because of the size of the images), so that each entry in the array is one image 
After that, the network gets built, trained and the model gets exported to a .keras file for inference. 
