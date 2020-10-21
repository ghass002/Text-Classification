# Text-Classification
building and training a MLP/ sepCNN model for text classification

## Algorithm for Data Preparation and Model Building
1. Calculate the number of samples/number of words per sample ratio.
2. If this ratio is less than 1500, tokenize the text as n-grams and use a simple multi-layer perceptron (MLP) model to classify them:
  1. Split the samples into word n-grams; convert the n-grams into vectors.
  2. Score the importance of the vectors and then select the top 20K using the scores.
  3. Build an MLP model.
3. If the ratio is greater than 1500, tokenize the text as sequences and use a sepCNN model to classify them :
  1. Split the samples into words; select the top 20K words based on their frequency.
  2. Convert the samples into word sequence vectors.
  3. If the original number of samples/number of words per sample ratio is less than 15K, using a fine-tuned pre-trained embedding with the sepCNN model will likely provide the best results.
4. Measure the model performance with different hyperparameter values to find
   the best model configuration for the dataset.
