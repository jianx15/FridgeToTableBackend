import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model

class TFModel:
    def __init__(self) -> None:
        self.df = pd.read_csv("../recipe_data/training.csv")
        self.tfidf_vectorizer = TfidfVectorizer()
        self.ingredients_tfidf = self.tfidf_vectorizer.fit_transform(self.df['Ingredients'])
        self.loaded_model = tf.keras.models.load_model('../recipe_data/tf_model.h5')
        self.label_encoder_foodtype = LabelEncoder()
        self.label_encoder_foodtype.fit(self.df['FoodType'])
        self.label_encoder_userid = LabelEncoder()
        self.label_encoder_userid.fit(self.df['UserId'])

    def recommend(self, user_id, food_type, ingredients):
        new_data = pd.DataFrame({
            'UserId': [user_id],
            'FoodType': [food_type],
            'Ingredients': [ingredients]
        })   

        new_data['FoodType'] = self.label_encoder_foodtype.transform(new_data['FoodType'])
        new_data['UserId'] = self.label_encoder_userid.transform(new_data['UserId'])
        new_data_tfidf = self.tfidf_vectorizer.transform(new_data['Ingredients'])
        
        prediction = self.loaded_model.predict(pd.concat([new_data['UserId'], new_data['FoodType'], pd.DataFrame(new_data_tfidf.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())], axis=1))
        return prediction[0][0]
