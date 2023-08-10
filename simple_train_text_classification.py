#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import texthero as hero

import numpy as np

from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_validate

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

import tqdm

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.utils.class_weight import compute_class_weight


# In[4]:



df = pd.read_table("trip_advisor_dataset.csv", sep=';')


# In[4]:


df_yelp = pd.read_table("manual_reviews.csv", sep=';')


# In[ ]:


df['dataset'] = 'TripAdvisor'

df_yelp['dataset'] = 'Yelp'


size_yelp, size_tripadvisor = len(df_yelp), len(df)

df = pd.concat([df_yelp, df])



# In[5]:


df = df[['text', 'trip type']]


# In[6]:


df['review_clean'] = hero.clean(df['text'])


# In[7]:


vectorizer = TfidfVectorizer()


# In[8]:


X = vectorizer.fit_transform(df['review_clean'])


# In[9]:


replace_classes = {trip: index for index, trip in enumerate(df['trip type'].unique())}


df['classes'] = df['trip type'].replace(replace_classes)


for train_dataset in ['Yelp', 'TripAdvisor']:
    
    if train_dataset == 'Yelp':
    
        x_train, y_train = X[: size_yelp], df['classes'][: size_yelp]
        x_test, y_test = X[-size_tripadvisor:], df['classes'][-size_tripadvisor: ]
    
    else:
    
        x_train, y_train = X[-size_tripadvisor:], df['classes'][-size_tripadvisor: ]
        x_test, y_test = X[: size_yelp], df['classes'][: size_yelp]
    
    
    # In[44]:
    
    complete_results = []
    
    for balanced in [True, False]:
    
    
    
        if balanced == True:
    
            models = {
                        
                        'LightGBM': LGBMClassifier(class_weight='balanced'),
                        'SVM': SVC(C=8.000e+00, kernel='linear', class_weight='balanced'),
                        'RF': RandomForestClassifier(max_depth=5, n_jobs=-1, class_weight='balanced'),
                        'GB': GradientBoostingClassifier(n_estimators=10, max_depth=3),
                        'XGBoost': XGBClassifier(n_jobs=-1),
                        'LogisticRegression': LogisticRegression(class_weight='balanced')
                    
                     }
    
    
            # In[45]:
    
            sample_weight = compute_class_weight(class_weight='balanced', classes=df['classes'].unique(), y=df['classes'])
    
            for label, class_name in enumerate(df['classes'].unique()):
    
                df.loc[df['classes'] == class_name, 'class_weight'] = sample_weight[label]
    
        else:
    
            df['class_weight'] = 1
    
            models = {
                        
                        'LightGBM': LGBMClassifier(),
                        'SVM': SVC(C=8.000e+00, kernel='linear'),
                        'RF': RandomForestClassifier(max_depth=5, n_jobs=-1),
                        'GB': GradientBoostingClassifier(n_estimators=10, max_depth=3),
                        'XGBoost': XGBClassifier(n_jobs=-1),
                        'LogisticRegression': LogisticRegression()
                    
                     }
    
    
        for model_name, model in tqdm.tqdm(models.items()):
    
            print("Model - ", model_name, balanced)
    
            error = False
    
            if model in ['XGBoost', 'GB']:
    
    
                try:
    
    
                    model.fit(x_train, y_train, sample_weight=df['class_weight'])
    
    
                    model =+ ' Balanced'
    
                except:
    
                    error = True
    
    
    
            if not (model in (['XGBoost', 'GB'])) or error == True:
    
    
                model.fit(x_train, y_train)
    
    
            prediction = model.predict(x_test)
    
    
            results = {
                        'model': model,
                        'balanced': balanced,
                        'f1-micro': f1_score(y_test, prediction, average='micro'),
                        'f1-macro': f1_score(y_test, prediction, average='macro'),
                        'f1-binary': f1_score(y_test, prediction, average='binary'), 
                        'accuracy': accuracy_score(y_test, prediction)
                      }
    
    
            df_results = pd.DataFrame([results])
    
    
            complete_results.append(df_results)
    
    
    # In[46]:
    
    
    df_complete = pd.concat(complete_results)
    
    
    df_complete.to_csv("train_" + train_dataset + ".csv", sep=';', index=False)
