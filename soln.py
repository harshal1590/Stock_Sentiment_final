import numpy as np 
import pandas as pd
import string
from tqdm import tqdm
import math,nltk
import re
import time
from sklearn import feature_extraction
from textblob import TextBlob 
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import losses
from tensorflow.keras import activations
import sklearn
from tensorflow.keras.layers import LSTM
from matplotlib import pyplot as plt
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
max_words = 5000
max_len = 200
# nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Word2vec
import gensim
# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools
import re
import string
import pickle
#from keras.models import load_model
#from tensorflow.keras.preprocessing.image import img_to_array
#from keras.models import load_model
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers

_wnl = nltk.WordNetLemmatizer()


data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin', names = ['polarity','id','date','query','user','text'])
data.head()
max_words = 5000
max_len = 200

data = data.sample(frac=1)
data = data[:200000]

data.drop(['date','query','user'], axis=1, inplace=True)

urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = '@[^\s]+'

stopword = set(stopwords.words('english'))
print(stopword)

def process_tweets(tweet):
  # Lower Casing
    tweet = tweet.lower()
    tweet=tweet[1:]
    # Removing all URls 
    tweet = re.sub(urlPattern,'',tweet)
    # Removing all @username.
    tweet = re.sub(userPattern,'', tweet) 
    #Remove punctuations
    tweet = tweet.translate(str.maketrans("","",string.punctuation))
    #tokenizing words
    tokens = word_tokenize(tweet)
    #Removing Stop Words
    final_tokens = [w for w in tokens if w not in stopword]
    #reducing a word to its word stem 
    wordLemm = WordNetLemmatizer()
    finalwords=[]
    for w in final_tokens:
      if len(w)>1:
        word = wordLemm.lemmatize(w)
        finalwords.append(word)
    return ' '.join(finalwords)

data['processed_tweets'] = data['text'].apply(lambda x: process_tweets(x))
print('Text Preprocessing complete.')


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data.processed_tweets)
model =load_model('model_new.h5')

def clean_tweet(tweet): 

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet): 

    sequence = tokenizer.texts_to_sequences([process_tweets(tweet)])
    test = pad_sequences(sequence, maxlen=max_len)
    pred = model.predict(test)
    return pred[0][0]



def get_tweet_sentiment_senti(tweet): 

    sequence = tokenizer.texts_to_sequences([process_tweets(tweet)])
    test = pad_sequences(sequence, maxlen=max_len)
    pred = model.predict(test)
    
    if pred[0][0] > 0.5:
        return 'positive'
    else:
        return 'negative'

# def processing(titles): 

#     tweets = [] 
#     try: 
#         fetched_tweets =  titles
#         for tweet in fetched_tweets: 
#             parsed_tweet = {} 
#             parsed_tweet['text'] = tweet
#             parsed_tweet['sentiment'] = get_tweet_sentiment(tweet) 
#             if tweet.retweet_count > 0: 
#                if parsed_tweet not in tweets: 
#                     tweets.append(parsed_tweet) 
#             else: 
#                 tweets.append(parsed_tweet) 
#         return tweets 
  
#     except tweepy.TweepError as e: 
#         print("Error : " + str(e))

root = Tk()  # Main window 
f = Frame(root)
frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root)
root.title("Stock Market Prediction With Sentiment Analysis")
root.geometry("1080x720")

canvas = Canvas(width=1080, height=250)
canvas.pack()
filename=('stock1.png')
load = Image.open(filename)
load = load.resize((1800, 250), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
#photo = PhotoImage(file='landscape.png')
load = Image.open(filename)
img.place(x=1, y=1)
#canvas.create_image(-80, -80, image=img, anchor=NW)


root.configure(background='Green')
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

firstname = StringVar()  # Declaration of all variables
lastname = StringVar()
id = StringVar()
dept = StringVar()
designation = StringVar()
remove_firstname = StringVar()
remove_lastname = StringVar()
searchfirstname = StringVar()
sevierity = StringVar()
sheet_data = []
row_data = []



def add_entries():  # to append all data and add entries on click the button
    a = " "
    f = sevierity.get()
    f1 = f.lower()
    l = lastname.get()
    l1 = l.lower()
    d = dept.get()
    d1 = d.lower()
    de = designation.get()
    de1 = de.lower()
    
    list1 = list(a)
    list1.append(f1)
    list1.append(l1)
    list1.append(d1)
    list1.append(de1)



def visualizations():
    import seaborn as sns
    sns.set_style('whitegrid')
    plt.style.use("fivethirtyeight")
    from datetime import datetime

    name=sevierity.get()
    filenames="Stocks_dataset/" + name + '_Stocks.csv'
    data=pd.read_csv(filenames)
    #Adj Close
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.77,bottom=0.28)


    plt.subplot(2, 2, 1)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"{name}")


    plt.subplot(2, 2, 2)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['Close'].plot()
    plt.ylabel('Close')
    plt.xlabel(None)
    plt.title(f"{name}")



    plt.subplot(2, 2, 3)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['High'].plot()
    plt.ylabel('High')
    plt.xlabel(None)
    plt.title(f"{name}")
    plt.show()



def senti_graph():
    name=sevierity.get()
    filenames="Tweets/" + name + '.csv'
    df = pd.read_csv(filenames)
    ptweets=[]
    ntweets=[]
    titles=  df['Text']


    ptweets =[tweet for tweet in titles if get_tweet_sentiment_senti(tweet) == 'positive']
    a=100*len(ptweets)/len(df)

    #ntweets =[tweet for tweet in titles if get_tweet_sentiment_senti(tweet) == 'negative']
    #b=100*len(ptweets)/len(df)

    b=100-a
    #b=100*len(ptweets)/len(df)


    e3.delete(0, END) #deletes the current value
    e3.insert(0, a)
    e4.delete(0, END) #deletes the current value
    e4.insert(0, b)
    # e5.delete(0, END) #deletes the current value
    # e5.insert(0, c)

    #pie chart
    labels = 'Possitive', 'Negative'
    sizes = [a,b]
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'possitive')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()




def click():
    name=sevierity.get()
    filenames= "Tweets/"+ name + '.csv'
    df = pd.read_csv(filenames)
    processed_senti=[]
    #titles= get_tweet_sentiment(df['Text'])
    titles = df['Text'].tolist()
    for tweet in titles:
        processed_senti.append(get_tweet_sentiment(tweet))

    df['sentiment']=processed_senti

    print(df)

    df = df.drop(['Text'],1)
    df['Date'] = pd.to_datetime(df['Date'])
    #df['Dates'] = df['Date'].dt.date()
    group = df.groupby('Date')
    sentiment_avg = group['sentiment'].mean()

    df_stock=pd.read_csv('Stocks_dataset/'+name+'_Stocks.csv')
    df_stock['sentiment_polarity']=sentiment_avg.values
    print(df_stock)

    X=df_stock[['sentiment_polarity','Open','High','Low','Adj Close']]
    Y=df_stock[['Close']]

    # X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33)

    length=(len(df_stock) / 100)
    length=round(length * 80)

    X_train=X[0:length] 
    X_test=X[length:]
    Y_train=Y[0:length] 
    Y_test=Y[length:]
    print(type(X_test))
    senti_polarity = X_test["sentiment_polarity"].mean()
    Open  = X_test["Open"].mean()
    High  = X_test["High"].mean()
    Low  = X_test["Low"].mean()
    AdjClose  = X_test["Adj Close"].mean()
    test_data = [senti_polarity,Open,High,Low,AdjClose]
    test_columns = ['sentiment_polarity','Open','High','Low','Adj Close']
    test_df = pd.DataFrame([test_data], columns=test_columns)
    print(type(test_df))

    y_test = Y_test


    from sklearn import preprocessing
    min_max_scalar=preprocessing.MinMaxScaler()
    X_train=min_max_scalar.fit_transform(X_train)
    X_test=min_max_scalar.fit_transform(X_test)
    Y_train=min_max_scalar.fit_transform(Y_train)
    Y_test=min_max_scalar.fit_transform(Y_test)

    

    #model
    model=Sequential()
    model.add(Dense(5,activation=activations.sigmoid,input_shape=(5,)))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(1,activation=activations.sigmoid))

    model.compile(optimizer='adam',loss=losses.mean_absolute_error)

    model.fit(X_train,Y_train,verbose=2,epochs=1000)
    y_pred = model.predict(X_test)

    y_pred=min_max_scalar.inverse_transform(y_pred)
    Y_test=min_max_scalar.inverse_transform(Y_test)
    
    # test_df=min_max_scalar.fit_transform(test_df)
    # predicted_price = model.predict(test_df)
    # predicted_price=min_max_scalar.inverse_transform(predicted_price)
    # print(predicted_price)


    # Y_test_ =  pd.concat([y_test, test_df])
    # Y_test_=min_max_scalar.fit_transform(Y_test_)
    predicted_price = y_pred[-8]
    e5.delete(0, END) #deletes the current value
    e5.insert(0, predicted_price)
    # predicted_price = model.predict(Y_test_)
    # predicted_price=min_max_scalar.inverse_transform(predicted_price)
    # print(predicted_price)


    date_list=df_stock['Date']
    pred=y_pred[-7:]
    orig=Y_test[-7:]
    dates=date_list[-7:]
    date_df=pd.DataFrame(dates,columns=['Date'])



    # Visualising the results
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()  
    plt.plot(orig, color = 'red', label = 'Real Stock Price')
    plt.plot(pred, color = 'blue', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction', fontsize=40)
    date_df.set_index('Date', inplace= True)
    date_df = date_df.reset_index()
    x=date_df.index
    labels = date_df['Date']
    plt.xticks(x, labels, rotation = 'vertical')
    for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks(): 
                tick.label1.set_fontsize(18)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('Stock Price', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})
    plt.show()



def clear_all():  # for clearing the entry widgets
    frame1.pack_forget()
    frame2.pack_forget()
    frame3.pack_forget()


label1 = Label(root, text="Stock Market Prediction With Sentiment Analysis")
label1.config(font=('Italic', 18, 'bold'), justify=CENTER, background="Yellow", fg="Red", anchor="center")
label1.pack(fill=X)


frame2.pack_forget()
frame3.pack_forget()


satisfaction_level = Label(frame2, text="Enter Query Text: ", bg="red", fg="Black")
satisfaction_level.grid(row=1, column=1, padx=10)
sevierity.set("Select Stock")
e1 = OptionMenu(frame2, sevierity, "Select Option", "AAPL", "AMZN", "FB", "INFY", "MSFT" ,"RELIANCE", "TCS", "TECHM")
e1.grid(row=1, column=2, padx=10)


button5 = Button(frame2, text="Submit", command=click)
button5.grid(row=1, column=3, pady=10,padx=10)


sentilabel = Label(frame2, text="View Sentiment: ", bg="red", fg="Black")
sentilabel.grid(row=2, column=1, padx=10)

button2 = Button(frame2, text="Sentiment",command=senti_graph)
button2.grid(row=2, column=2, pady=10,padx=10)


visualabel = Label(frame2, text="View Visualization: ", bg="red", fg="Black")
visualabel.grid(row=3, column=1, padx=10)

button2 = Button(frame2, text="Visualization",command=visualizations)
button2.grid(row=3, column=2, pady=10,padx=10)


predlabel = Label(frame2, text="Predicted Pice: ", bg="red", fg="Black")
predlabel.grid(row=4, column=1, padx=10)
e5 = Entry(frame2)
e5.grid(row=4, column=2, padx=10, pady=10)


label1 = Label(frame1, text="Possitive Tweets ", bg="red", fg="Black")
label1.grid(row=4, column=1, padx=10, pady=10)
e3 = Entry(frame1)
e3.grid(row=5, column=1, padx=10, pady=10)

label2 = Label(frame1, text="Negative Tweets ", bg="red", fg="Black")
label2.grid(row=4, column=2, padx=10, pady=10)
e4 = Entry(frame1)
e4.grid(row=5, column=2, padx=10, pady=10)




frame2.configure(background="Red")
frame2.pack(pady=10)

frame1.configure(background="Red")
frame1.pack(pady=10)

root.mainloop()






















