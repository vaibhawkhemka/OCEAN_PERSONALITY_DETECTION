# OCEAN_PERSONALITY_DETECTION
# REQUIREMENTS:
•nltk (TweetTokenizer)
•Keras
•Tensorflow
•numpy
•scipy
•itertools

# PREPROCESSING:
•“Data/data_preprocessing/data_handler.py” prepares the data for training and output the preprocessed data as “Text_files/train/train_data.txt” 
•“Data/data_preprocessing/test_handler.py”  gives output as  “Text_files/test/test_data.txt”. 

# TRAINING:
•For Training, Run OCEAN_MODEL.py after uncommenting “tr = train_model(..) “ [Line 322] and commenting “t=test_model(..)”[Line 325], “t.load_trained_model(..)”[Line   326] ,”t.test_predict”[Line 327]
•Training data after preprocessing used – “Text_files/Train/train_data.txt”.
•Personality Trait:
0:  Extroversion
1:  Neuroticism
2:  Agreeableness
3:  Conscientiousness
4:  Openness

# TESTING:
•For testing, run “OCEAN_PERSONALITY_DETECTION/predict_type.py”  and it will give multi-classification output.
•It will give output with information extracted as “OCEAN_PERSONALITY_DETECTION/information.txt”
•And Prediction for all test cases as “OCEAN_PERSONALITY_DETECTION/predictions.txt”

