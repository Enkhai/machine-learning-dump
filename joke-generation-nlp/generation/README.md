# Generation
GeneratedJokes.csv: CSV dataframe containing hyperparameters for generation, a seed text and a generated output text.
<br>Based on seed.csv.

generate.py: Contains generative code.
<br>A trained model, an index2word dictionary, a word2index dictionary and a seed dataframe are loaded to generate text. 
<br>The seed dataframe contains hyperparameters for generation and seed text for every row.
<br>Generation is also possible by using a generateOnce function that will generate text once using user-defined hyperparameters and seed text.

seed.csv: CSV dataframe containing hyperparameters for generation and a seed text.
<br>GeneratedJokes.csv and seed.csv share a similar structure, with GeneratedJokes.csv also containing generated text.

train.py: Contains data preprocessing, model architecture and training code.
<br>Once should first run the training code to produce the index2word dictionary, word2index dictionary and generated model before running the generative code.
<br>A system with an Nvidia GPU and tensorflow-gpu installed is required. 