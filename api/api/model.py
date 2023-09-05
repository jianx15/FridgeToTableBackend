import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Model:
    def __init__(self) -> None:
        self.df = pd.read_csv("../recipe_data/final_data.csv")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.ingredients_matrix = self.vectorizer.fit_transform(self.df["Ingredients"]).toarray()
        
    def get_recommendation(self, user_input) -> None:
        user_input_vector = self.vectorizer.transform([user_input])
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(user_input_vector, self.ingredients_matrix)

        # Get the index of the most similar recipe
        most_similar_index = cosine_similarities.argmax()

        # Get the recommended recipe
        recommended_recipe = self.df.iloc[most_similar_index]

        return recommended_recipe.to_json(orient='records')