# OCEAN_PERSONALITY_DETECTION
# REQUIREMENTS:
•nltk (TweetTokenizer)
•Keras
•Tensorflow
•numpy
•scipy
•itertools

# PREPROCESSING:
• “DATA/data_preprocessing/data_handler.py” prepares the data for training and output the preprocessed data as “TEXT_FILES/TRAIN/train_data.txt” 
• “DATA/test_handler.py”  gives output as  “TEXT_FILES/TEXT/information.txt”. 

# TRAINING:
• For Training, Download http://nlp.stanford.edu/data/glove.6B.zip in "/Data/" and set glove_path = "/DATA/glove.6B.300d.txt" after zipping the file in "/DATA/" 
• Run OCEAN_MODEL.py after uncommenting “tr = train_model(..) “ [Line 322] and commenting “t=test_model(..)”[Line 325], “t.load_trained_model(..)”[Line   326]
  ,”t.test_predict”[Line 327]
• Training data after preprocessing used – “TEXT_FILES/TRAIN/train_data.txt”.
• The Model will be stored in “TEXT_FILES/TEXT_MODEL/weights/model.json” and weights has to be stored to “TEXT_FILES/TEXT_MODEL/weights/model.json.hdf5” using   https://drive.google.com/file/d/1JJhd1svjXYwbFngG7kqw_DmEzY-HG_Yg/view?usp=sharing 
• Personality Trait:
0:  Extroversion
1:  Neuroticism
2:  Agreeableness
3:  Conscientiousness
4:  Openness

# TESTING:
•Model has to be loaded before training.
•For testing, run “OCEAN_PERSONALITY_DETECTION/DATA/predict_type.py”  and it will give multi-classification output.
•It will give output with information extracted as “OCEAN_PERSONALITY_DETECTION/TEXT_FILES/TEXT/information.txt”
•And Prediction for all test cases as “OCEAN_PERSONALITY_DETECTION/DATA/predictions.txt”

