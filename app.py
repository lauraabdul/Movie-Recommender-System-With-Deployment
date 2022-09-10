# Import the Libraries
from re import M
import string
import flask
import pandas as pd 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, redirect, render_template, request

import os

from utils import get_recommendation ,cosine_sim

app = Flask(__name__)

df=pd.read_csv("movies__final_data.csv")

# create array with all movie titles
all_titles = [df['title'][i] for i in range(len(df['title']))]


@app.route('/')
def home():
    return render_template('index.html')

# Route for Predict page


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # while prediction

        m_name = " ".join(flask.request.form['movie_name'].split())
        
        #lis=m_n.split()
        #for i in range(len(lis)):
            #if lis[i] not in ['of','the','on','a','an']:
                #lis[i]=string.capwords(lis[i])

        #m_name=' '.join(lis)
        if m_name not in all_titles:
            return(flask.render_template('notfound.html')) #######
        else:
            result_final = get_recommendation(m_name)
            names = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i])
            return flask.render_template('predict.html',movie_names=names)
    else:
        return flask.render_template('predict.html') 




# Run the App from the Terminal
if __name__ == '__main__':
    app.run(debug=True)