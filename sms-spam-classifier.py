#!/usr/bin/env python
# coding: utf-8

# # SMS Spam Classifier | End to End Project | Streamlit Share Deployment
# 

# 
# ## Introduction: Building an Intelligent Spam Classifier for Emails and SMS
# 
# #### In today's fast-paced digital world, the volume of online communication has skyrocketed. Unfortunately, this surge has brought with it an unwelcome guest - spam! Whether it's flooding our email inboxes or bombarding us with unwanted text messages, spam can be annoying, time-consuming, and even potentially harmful.
# 
# #### But fear not! We have an ingenious solution in store for you - an intelligent Spam Classifier powered by state-of-the-art artificial intelligence! This end-to-end project aims to develop an efficient system capable of accurately identifying spam emails and SMS, saving you precious time and ensuring you never miss important messages again.
# 
# ## The Dataset: Unveiling the Secrets of Spam and Ham
# 
# #### Behind every great AI lies a carefully curated dataset, and our Spam Classifier is no exception! For this project, we have obtained a rich and diverse dataset containing thousands of emails and SMS messages, meticulously labeled as either "spam" or "ham" (non-spam).
# 
# #### We take dataset from the kaggle. You can see from the below link.
# ### https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# 
# #### The dataset covers a wide range of spamming techniques and tricks used by spammers to infiltrate our inboxes. It includes various types of unsolicited promotional content, fraudulent schemes, phishing attempts, and more. On the other hand, the "ham" messages represent legitimate and desired communication that we cherish, such as personal messages, work-related updates, and newsletters we willingly subscribed to.
# 
# ## Our Approach: Unleashing the Power of Machine Learning
# 
# #### Now comes the exciting part - our approach to building the Spam Classifier! We'll employ cutting-edge machine learning techniques to transform this raw dataset into a powerful predictive model. We'll dive deep into Natural Language Processing (NLP) algorithms, feature engineering, and smart model selection to create a robust and accurate classifier.
# 
# #### The heart of our system will be a sophisticated neural network, trained on this dataset, learning from the patterns and characteristics that distinguish spam from genuine messages. By leveraging the power of AI, our Spam Classifier will continuously adapt and improve its accuracy over time, making it a formidable guardian against spam attacks.
# 
# ## Streamlit Deployment: Making it User-Friendly and Accessible
# 
# #### But wait, we don't want this powerful tool to be confined to the hands of tech experts! We believe that protecting against spam should be easy and accessible to everyone. That's why we're employing Streamlit, a user-friendly Python library for creating web applications, to deploy our Spam Classifier.
# 
# #### With Streamlit, you'll experience a seamless and intuitive interface. Just input an email or SMS message, click the magic button, and our intelligent classifier will swiftly determine whether it's spam or ham! Say goodbye to the stress of dealing with spam - we've got you covered!
# 
# ## Join Us on this Spam-Busting Journey!
# 
# #### So, are you ready to take control of your inbox and SMS inbox once and for all? Let's embark on this spam-busting journey together! Our intelligent Spam Classifier, fortified with the latest AI techniques and the wisdom of a carefully curated dataset, is here to make your digital life safer, smoother, and spam-free. Let's put an end to spam together!
# 
# 
# 
# 
# 

# In[131]:


import pandas as pd


# In[132]:


df = pd.read_csv('spam.csv')


# In[133]:


df.head()


# #### You may sometime face error while loading data as dataframe from csv file as not UTF-8 encoded. You can fine online converter to change not incoded to coded.

# #### TF-8 (Unicode Transformation Format - 8-bit) is a character encoding standard for Unicode, which is a universal character set that aims to encompass all characters from all human languages and symbols used in modern computing. UTF-8 is a variable-width encoding, meaning it can represent Unicode characters using one to four bytes, depending on the character's code point.
# 
# ## UTF-8 Encoded Data:
# #### In UTF-8 encoded data, text characters are represented using a series of bytes, and each character is mapped to a specific sequence of bytes. UTF-8 can represent characters from various languages, including English, Chinese, Arabic, and others.
# #### UTF-8 is widely used because it allows efficient representation of ASCII characters (which are commonly used in the English language) using just one byte, while still supporting the full range of Unicode characters.
# 
# ## Not Encoded Data:
# #### When data is said to be "not encoded," it means that the text characters are represented directly as they are, without any specific encoding applied. This typically implies that the text data is encoded using the system's default encoding (e.g., on Windows, it is often 'cp1252', while on Linux, it is typically 'utf-8').
# #### In such cases, if the file contains characters that are not compatible with the default encoding, a UnicodeDecodeError can occur when trying to interpret the data.
# 
# ## Importance of Encoding:
# #### Character encoding is essential because it defines the mapping between bytes and characters, allowing computers to correctly interpret and display text data.
# #### Using a standardized encoding like UTF-8 ensures that text data can be exchanged and processed across different platforms and systems without losing information or introducing errors.
# 
# ## Jupyter Notebook and Encoded Data:
# #### Jupyter Notebook may ask for encoded data when reading files or processing text data to ensure that the data is interpreted correctly. By explicitly specifying the encoding, you help Jupyter Notebook understand how to interpret the bytes in the file and map them to Unicode characters.
# #### If Jupyter Notebook doesn't know the encoding and encounters non-ASCII characters or characters from different languages, it may fail to decode the data correctly, leading to the UnicodeDecodeError.
# 
# #### In summary, using a proper encoding like UTF-8 is crucial to handle text data that contains characters from various languages and scripts, ensuring that the data is read and processed correctly without errors.

# In[134]:


df.sample(5)


# In[135]:


df.shape


# ## Here's a brief explanation of each stage in your SMS spam classifier project:
# 
# ### Data Cleaning:
# #### In this stage, you'll clean and preprocess the raw SMS data to remove any irrelevant or redundant information, handle missing values, and correct errors. Data cleaning ensures that the dataset is consistent and ready for further analysis.
# 
# ### Exploratory Data Analysis (EDA):
# #### EDA involves exploring the cleaned dataset to gain insights and understand the distribution of spam and non-spam (ham) messages. You'll visualize the data, analyze patterns, and extract useful features that can help in building an effective spam classifier.
# 
# ### Text Preprocessing:
# #### Text preprocessing prepares the text data for machine learning models. It involves converting the text into a numerical format, tokenization, removing stopwords, stemming or lemmatization, and transforming the text into a format suitable for model training.
# 
# ### Model Building:
# #### In this stage, you'll develop a machine learning model using the preprocessed data. Commonly used algorithms for text classification include Naive Bayes, Support Vector Machines (SVM), and deep learning models like Recurrent Neural Networks (RNNs) or Transformer-based models.
# 
# ### Evaluation:
# #### After training the model, you'll evaluate its performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrix. Evaluation helps you understand how well your model is performing and identify areas for improvement.
# 
# ### Improvement:
# #### Based on the evaluation results, you might fine-tune your model by adjusting hyperparameters, using different algorithms, or modifying the feature engineering process. The goal is to improve the model's performance and reduce false positives or false negatives.
# 
# ### Website:
# #### Creating a website allows users to interact with your SMS spam classifier. You can build a simple user interface where users can input text messages, and the classifier will predict whether it's spam or not.
# 
# ### Deployment:
# #### Deployment involves making your trained model accessible to users. You can deploy the model and website on cloud platforms or web servers, making it available for real-world use.
# 
# #### Remember, throughout the project, it's essential to iterate and refine each stage to achieve the best results. Good luck with your SMS spam classifier project!

# ## 1. Data Cleaning

# #### I want to know whether the other columns has any works on the modelling.

# In[136]:


df.info()


# #### As the data in last 3 columns are very less, let's drop them.

# In[137]:


df.drop(columns = [ 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)


# In[138]:


df.sample(5)


# #### Let's us change the column name of v1 and v2.

# In[139]:


df.rename(columns = {'v1' : 'target' , 'v2' :'text'}, inplace = True)
df.sample(5)


# #### I want to convert the target from word to numerical value. I mean i want to convert ham and spam using label encoder.

# In[140]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[141]:


df['target'] = encoder.fit_transform(df['target'])


# #### The ham is assigned 0 and spam to 1

# In[142]:


df.head(5)


# In[143]:


# Missing value
df.isnull().sum()


# In[144]:


# Check for duplicate values
df.duplicated().sum()


# #### There are 403 duplicate values and we need to remove them

# In[145]:


df = df.drop_duplicates(keep='first')


# In[146]:


df.duplicated().sum()


# #### Hence we have no duplicate datas now.

# In[147]:


df.shape


# ## 2. EDA

# #### I  want to know how much percentage of sms are spam and how much is ham

# In[148]:


df['target'].value_counts()


# In[149]:


import matplotlib.pyplot as plt


# In[150]:


plt.pie(df['target'].value_counts(),labels = ['ham','spam'],autopct="%0.2f")
plt.show()


# #### In data visualization libraries like Matplotlib and Pandas, the autopct parameter is used to format the display of percentage values in pie charts or other types of charts where percentages are relevant.
# 
# #### Specifically, autopct="%0.2f" is a formatting specification that tells the chart to display the percentage values with two decimal places (rounded to two decimal points). The "%0.2f" format string consists of the following components:
# 
# #### %: This is a special character used to indicate that the value to be inserted will be formatted.
# #### 0: This is the padding specifier, indicating that if the number has fewer digits than specified (in this case, two decimal places), it will be zero-padded on the left.
# #### .2: This is the precision specifier, indicating the number of decimal places to display (in this case, two decimal places).
# #### f: This is the type specifier, indicating that the number to be formatted is a floating-point number.

# #### From the pie chart we can conclude that data is imbalanced.

# #### Now we need to analyse how much of our data is alphabet, words and sentences.

# In[151]:


import nltk


# In[152]:


nltk.download('punkt')


# 
# #### The command nltk.download('punkt') is used in the NLTK (Natural Language Toolkit) library, a popular Python library for working with human language data. Specifically, this command is used to download the "punkt" data package, which is a pre-trained tokenizer model provided by NLTK.
# 
# ### Summary of nltk.download('punkt'):
# 
# ### Tokenization: 
# #### Tokenization is the process of splitting a text into individual words or sentences. It is a fundamental step in natural language processing tasks.
# 
# ### 'punkt' data package: 
# #### The 'punkt' package contains pre-trained models for tokenization in various languages. These models are trained on large corpora of text and can accurately segment text into words or sentences.
# 
# #### NLTK download function: 
# ### nltk.download() is a function provided by the NLTK library to download specific data packages needed for various NLP tasks.
# 
# #### One-time download: 
# ### Running nltk.download('punkt') once in your code will download the 'punkt' data package to your system. After that, you can use the tokenization functionality provided by the package without needing to download it again.
# 
# ### Benefits: 
# #### By downloading the 'punkt' package, you can access high-quality and language-specific tokenizers for many languages, making it easier to process and analyze text in your NLP applications.
# 
# #### Once you have downloaded the 'punkt' package, you can utilize its tokenization capabilities through the NLTK library and start tokenizing text into words or sentences as needed for your NLP tasks

# In[153]:


df['num_characters']=df['text'].apply(len)


# In[154]:


df.head()


# #### To find the number of text

# In[155]:


df['text'].apply(lambda x:nltk.word_tokenize(x))


# In[156]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# #### nltk.word_tokenize() is another function provided by the Natural Language Toolkit (NLTK) in Python. It is used for word tokenization, which means breaking a piece of text into individual words or tokens.
# 
# #### The nltk.word_tokenize() function takes a text string as input and returns a list of words present in that text. It uses various rules and tokenization models to handle different cases, such as punctuation, contractions, and special characters

# In[157]:


df.head(5)


# In[158]:


df['text'].apply(lambda x:nltk.sent_tokenize(x))


# In[159]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# #### nltk.sent_tokenize() is a function provided by the Natural Language Toolkit (NLTK), which is a popular library for working with human language data in Python. This function is used for sentence tokenization, which means breaking a text into individual sentences.
# 
# #### The nltk.sent_tokenize() function takes a piece of text as input and returns a list of sentences present in that text. It uses pre-trained models and rules to identify sentence boundaries accurately, even in the presence of tricky cases like abbreviations, punctuation, or other sentence-ending marks.

# In[160]:


df.head(5)


# In[161]:


df[['num_characters','num_words','num_sentences']].describe()


# In[162]:


df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[163]:


df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[164]:


import seaborn as sns


# In[165]:


plt.figure(figsize = (12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'], color = 'red')


# #### From the graph we can clearly see that ham messages have lesser words in comparison to the spam messages

# In[166]:


plt.figure(figsize = (12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color = 'red')


# #### This shows that the spam messages have more words in comparison to the ham.

# In[167]:


plt.figure(figsize = (12,6))
sns.histplot(df[df['target'] == 0]['num_sentences'])
sns.histplot(df[df['target'] == 1]['num_sentences'], color = 'red')


# #### To see the relationship between data columns we can use pairplot.

# In[168]:


sns.pairplot(df, hue = 'target' )


# #### In data visualization libraries like seaborn and pandas, the hue parameter is used to introduce an additional categorical variable into a plot, which allows you to differentiate the data points based on this variable. It is especially useful when you want to visualize the relationship between multiple variables and observe how they vary concerning one another.

# #### This doesnot clearly suggest much on relationship between datas but we can clearly see some outliers.

# #### Let's see some correlation between the datas.

# In[169]:


df.corr()


# In[170]:


sns.heatmap(df.corr(), annot = True)


# #### In Python, the annot=True parameter is typically used in data visualization libraries, specifically when creating heatmaps with the seaborn or pandas libraries. Heatmaps are a graphical representation of data where values are displayed as colors on a 2D matrix, and annot=True adds annotations (text labels) to each cell of the heatmap, displaying the actual data values associated with that cell.

# #### Since num_characters, num_words and num_sentences are highly correlated hence we will remove other column than num_characters since it has highest correlation coefficient with target as 0.38.

# ## 3. Data Preprocessing
# #### Lower case
# #### Tokenization
# #### Removing special characters
# #### Removing stop words and punctuation
# #### Stemming

# In[171]:


from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords.words('english')


# In[172]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[173]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    #data cannot be copied as text = y always be careful
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
                
                
    return " ".join(y)
    


# #### The isalnum() function is a method in Python that belongs to the str class. It is used to check whether a given string contains only alphanumeric characters or not. Alphanumeric characters are letters (both uppercase and lowercase) and digits (0-9).

# In[174]:


import string
string.punctuation


# In[175]:


transform_text('%%%your name is shyam hari loving love')


# In[176]:


df['text'].apply(transform_text)


# In[177]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[178]:


df.head()


# #### The line of code from wordcloud import WordCloud is an import statement in Python that allows you to use the WordCloud class from the wordcloud library/module. The WordCloud class is typically used for generating word clouds, which are visual representations of the frequency of words in a given text.

# In[179]:


pip install wordcloud


# In[180]:


from wordcloud import WordCloud
wc = WordCloud(width =500, height = 500, min_font_size = 10, background_color = 'white')


# In[181]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep = " "))


# In[182]:


plt.figure(figsize = (12,6))
plt.imshow(spam_wc)


# #### A word cloud is a popular data visualization technique used to represent the frequency of words in a given text. It visually displays words in different sizes, with more frequent words appearing larger and bolder than less frequent ones. Word clouds are often used to gain insights into the most important or commonly occurring words in a text corpus and to provide a quick overview of the textual data.

# #### The .str.cat() method in pandas is used to concatenate strings in a Series element-wise using a specified separator. It is similar to the .join() method, but it operates on a Series of strings rather than on lists or tuples.

# #### plt.imshow() is a function provided by the matplotlib.pyplot module in Python. It is commonly used to display images, 2D arrays, or heatmaps as visualizations. This function is part of the matplotlib library, which is widely used for creating static, interactive, and animated visualizations in Python.

# In[183]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep = " "))


# In[184]:


plt.figure(figsize = (12,6))
plt.imshow(ham_wc)


# #### We want to know the top 30 words of both spam and ham sms

# In[185]:


df[df['target'] == 1]['transformed_text'].tolist()


# In[186]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[187]:


spam_corpus


# In[188]:


len(spam_corpus)


# In[189]:


from collections import Counter
Counter(spam_corpus).most_common(30)


# In[190]:


from collections import Counter
pd.DataFrame(Counter(spam_corpus).most_common(30))[0]
                                                                                                        


# In[191]:


pd.DataFrame(Counter(spam_corpus).most_common(30))[1]


# In[192]:


sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation = 'vertical')
plt.show()


# #### The code snippet you provided uses the Counter class from the collections module in Python to count the occurrences of elements in the spam_corpus iterable and then returns the 30 most common elements along with their respective counts.

# In[193]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[194]:


ham_corpus


# In[195]:


len(ham_corpus)


# In[196]:


from collections import Counter
Counter(ham_corpus).most_common(30)


# In[197]:


sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation = 'vertical')
plt.show()


# ## 4. Model Building

# In[267]:


from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features = 3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[269]:


X.shape


# In[270]:


y = df['target'].values


# In[271]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split


# In[272]:


gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()


# In[273]:


X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = 2)


# In[274]:


mnb.fit(X_train, y_train)
y_pred3 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test,y_pred3))


# In[199]:


X = cv.fit_transform(df['transformed_text']).toarray()


# #### In scikit-learn, a popular machine learning library for Python, the sklearn.naive_bayes module provides several implementations of the Naive Bayes algorithm. Naive Bayes is a family of probabilistic algorithms based on Bayes' theorem, commonly used for classification tasks. The three main implementations in scikit-learn are Gaussian Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes. Let's explore each one in detail:
# 
# ### Gaussian Naive Bayes (GaussianNB):
# #### Gaussian Naive Bayes is used for classification when the features follow a Gaussian (normal) distribution. It assumes that the features are continuous numeric variables. It's commonly used when dealing with real-valued features.
# #### The key assumption in Gaussian Naive Bayes is that the likelihood of each feature belonging to a particular class follows a Gaussian distribution, and the class probability is estimated using maximum likelihood estimation.
# 
# ### Multinomial Naive Bayes (MultinomialNB):
# #### Multinomial Naive Bayes is suitable for classification tasks with discrete features, typically representing counts or frequencies of events. It is commonly used for text classification, where the features are word counts or term frequencies.
# #### In Multinomial Naive Bayes, the likelihood of each feature belonging to a particular class is modeled using a multinomial distribution, and the class probability is estimated using maximum likelihood estimation.
# 
# ### Bernoulli Naive Bayes (BernoulliNB):
# #### Bernoulli Naive Bayes is suitable for binary classification tasks where the features are binary, i.e., each feature can take only two values (0 or 1). It's often used for text classification tasks where the presence or absence of words is the main feature.
# #### In Bernoulli Naive Bayes, the likelihood of each feature belonging to a particular class is modeled using a Bernoulli distribution, and the class probability is estimated using maximum likelihood estimation.
# 
# #### In all three cases, you can use the trained model to make predictions on new data using the predict() method. The choice of which Naive Bayes implementation to use depends on the nature of your data and the assumptions about its distribution. Experimenting with different implementations is recommended to find the best one for your specific problem.

# #### The sklearn.metrics module in scikit-learn provides several functions to evaluate the performance of machine learning models. The functions you mentioned (accuracy_score, confusion_matrix, and precision_score) are commonly used metrics for assessing the performance of classification models. Let's briefly explain each of them:
# 
# ### accuracy_score(y_true, y_pred):
# #### The accuracy_score function is used to calculate the accuracy of a classification model's predictions. It compares the predicted labels (y_pred) to the true labels (y_true) and computes the proportion of correctly predicted instances over the total number of instances.
# #### The y_true argument should be the true labels (ground truth) of the data, and y_pred should be the predicted labels obtained from your model. The function returns a value between 0 and 1, where 1 represents perfect accuracy.
# 
# ### confusion_matrix(y_true, y_pred):
# #### The confusion_matrix function is used to create a confusion matrix, which is a table that shows the true positives, true negatives, false positives, and false negatives of a classification model's predictions.
# #### The y_true argument should be the true labels (ground truth) of the data, and y_pred should be the predicted labels obtained from your model. The confusion_matrix function returns a 2x2 array (for binary classification) or an NxN array (for multi-class classification) containing the counts of true positive, false positive, true negative, and false negative predictions.
# 
# #### precision_score(y_true, y_pred):
# #### The precision_score function is used to compute the precision of a classification model's predictions. Precision measures the proportion of true positive predictions (correctly predicted positive instances) over the total number of positive predictions made by the model.
# #### The y_true argument should be the true labels (ground truth) of the data, and y_pred should be the predicted labels obtained from your model. The function returns a value between 0 and 1, where 1 represents perfect precision.
# 
# #### These metrics are essential for evaluating the performance of classification models and understanding their strengths and weaknesses. When using scikit-learn, these functions can be used alongside your model to assess how well it generalizes to unseen data and to fine-tune its parameters if necessary.

# In[209]:


gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test,y_pred1))


# In[210]:


bnb.fit(X_train, y_train)
y_pred2 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test,y_pred2))


# In[211]:


mnb.fit(X_train, y_train)
y_pred3 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test,y_pred3))


# #### Let us use tf-idf. TF-IDF: The product of TF and IDF, which combines both metrics to provide a measure of the importance of a word within a specific document and across the entire document collection.
# #### TF-IDF = TF * IDF
# 
# 

# In[261]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()


# In[262]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[214]:


X.shape


# In[215]:


gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test,y_pred1))


# In[216]:


bnb.fit(X_train, y_train)
y_pred2 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test,y_pred2))


# In[217]:


mnb.fit(X_train, y_train)
y_pred3 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test,y_pred3))


# #### Let's go for tfidf with MNB

# In[218]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[219]:


lrc = LogisticRegression(solver = 'liblinear',penalty = 'l1')
svc = SVC(kernel = 'sigmoid', gamma = 1)
mnb = MultinomialNB ()
dtc = DecisionTreeClassifier(max_depth = 5)
knc = KNeighborsClassifier()
abc = AdaBoostClassifier(n_estimators = 50 , random_state = 2)
bc = BaggingClassifier(n_estimators = 50, random_state =2)
etc = ExtraTreesClassifier(n_estimators = 50, random_state =2)
gbdt = GradientBoostingClassifier(n_estimators = 50, random_state =2)
xgb = XGBClassifier(n_estimators = 50, random_state =2)
rfc = RandomForestClassifier(n_estimators = 50 , random_state = 2)


# ## 5. Evaluation

# In[220]:


clfs = {
    'LRC' : lrc,
    'SVC' : svc,
    'MNB' : mnb,
    'DTC' : dtc,
    'KNC' : knc,
    'ABC' : abc,
    'BC'  :bc,
    'ETC' : etc,
    'GBDT' : gbdt,
    'XGB' :xgb,
    'RFC' : rfc
    
}


# In[221]:


def train_classifier(clf, X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred =  clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[222]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[223]:


accuracy_scores = []
precision_scores = []
for name,clf in clfs.items():
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    print("for",name)
    print("accuracy :",current_accuracy)
    print("precision :",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[224]:


performance_df = pd.DataFrame({'Algorithm': clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores})


# In[225]:


performance_df


# In[226]:


performance_df.sort_values(by = 'Accuracy', ascending = False)


# In[227]:


performance_df1 = pd.melt(performance_df , id_vars = 'Algorithm')


# #### The pd.melt() function is a method provided by the Pandas library, which is a powerful and popular Python library for data manipulation and analysis. The melt() function is used to transform (or "unpivot") a DataFrame from a wide format to a long format, making it easier to analyze and work with the data.
# 
# #### The basic idea of melting a DataFrame is to convert columns into rows while keeping other columns as identifiers. This can be particularly useful when you have data stored in a wide format, where different columns represent different categories or variables, and you want to reorganize the data into a long format to perform various analyses or visualizations.
# 
# 

# In[228]:


performance_df1


# In[229]:


sns.catplot(x = 'Algorithm', y = 'value', hue = 'variable', data = performance_df1, kind = 'bar', height = 5)
plt.ylim(0.5,1.0)
plt.xticks(rotation = 'vertical')
plt.show()


# ## 6. Model Improvement

# In[230]:


#### Let us consider the max_features to 3000 to the tfidf model


# In[231]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 3000)


# In[232]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[233]:


X.shape


# In[234]:


accuracy_scores = []
precision_scores = []
for name,clf in clfs.items():
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    print("for",name)
    print("accuracy :",current_accuracy)
    print("precision :",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[235]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000' : accuracy_scores,'Precision_max_ft_3000' : precision_scores}).sort_values('Precision_max_ft_3000',ascending = False) 


# In[236]:


temp_df


# In[237]:


temp_df = performance_df.merge(temp_df, on = 'Algorithm')


# In[238]:


temp_df


# #### Let us scale X

# In[239]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[240]:


X.shape


# In[241]:


y = df['target'].values


# In[242]:


y


# In[243]:


from sklearn.model_selection import train_test_split


# In[244]:


X_train,X_test,y_train , y_test= train_test_split(X,y,test_size = 0.2,random_state = 2)


# In[245]:


accuracy_scores = []
precision_scores = []
for name,clf in clfs.items():
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    print("for",name)
    print("accuracy :",current_accuracy)
    print("precision :",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[246]:


new_df_scaled = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling' : accuracy_scores,'Precision_scaling' : precision_scores}).sort_values('Precision_scaling',ascending = False) 


# In[247]:


new_df_scaled = new_df_scaled.merge(temp_df, on = 'Algorithm')


# In[248]:


new_df_scaled


# ## Voting Classifier
# #### The Voting Classifier is a type of ensemble learning method in machine learning. It combines the predictions of multiple individual classifiers (also known as "base classifiers" or "estimators") to make a final prediction. The idea behind ensemble methods like the Voting Classifier is that combining the predictions of different models can often lead to better and more robust performance than using a single model.
# 
# #### The Voting Classifier in scikit-learn, a popular Python library for machine learning, is implemented using the VotingClassifier class. It allows you to combine different machine learning algorithms, such as decision trees, support vector machines, logistic regression, etc., into a single ensemble model.
# 
# #### There are two main types of voting in the Voting Classifier:
# 
# ### Hard Voting: 
# #### In hard voting, each base classifier's prediction is treated equally, and the final prediction is based on the majority vote (the most common predicted class). In the case of a tie, the class with the smallest index is chosen.
# 
# ### Soft Voting: 
# #### In soft voting, the classifiers' predicted probabilities for each class are averaged, and the class with the highest average probability is chosen as the final prediction. Soft voting requires classifiers to support the predict_proba method, which outputs the class probabilities.

# In[249]:


svc = SVC(kernel = 'sigmoid', gamma = 1, probability = True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50,random_state = 2)
from sklearn.ensemble import VotingClassifier


# In[250]:


voting = VotingClassifier(estimators= [('svm',svc),('nb',mnb),('et',etc)], voting = 'soft')


# In[251]:


voting.fit(X_train,y_train)


# In[252]:


VotingClassifier(estimators=[('svm',
                            SVC(gamma = 1.0, kernel='sigmoid',
                               probability = True)),
                             ('nb',MultinomialNB()),
                            ( 'et',
                             ExtraTreesClassifier(n_estimators=50,random_state =2))],
                 voting='soft')
                 
                             


# In[253]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# #### Applying stacking

# In[254]:


estimators = [('svc',svc),('nb',mnb),('et',etc)]
final_estimator = RandomForestClassifier()


# In[255]:


from sklearn.ensemble import StackingClassifier


# In[256]:


clf = StackingClassifier(estimators = estimators, final_estimator = final_estimator)


# In[257]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test,y_pred))
print("Prediction",precision_score(y_test,y_pred))


# In[258]:


print('hello')


# In[275]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# ### Now let us use PyCharm to deploy app using streamlit. We will publish our page through streamlit share using github repository.

# #### After running the codes above we will get two files vectorizer.pkl and model.pkl and will paste it on the project directory.
# #### Now we will create a new files and name it as app.py. In that file we will do coding for deployment.
# #### import streamlit
# import streamlit as st
# #### Now we need to download streamlit using streamlit inside the pycharm
# pip install streamlit
# 
# #### Now we will download the streamlit through terminal
# pip install streamlit
# 
# #### let us import stramlitas 
# import streamlit as st
# 
# #### To make our job easy let us go to stramlit documentation to see the different features.
# 
# #### using pickle to transfer datas 
# import pickle 
# 
# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# 
# model = pickle.load(open('model.pkl','rb'))
# 
# st.title('SMS Spam Classifier')
# 
# input_sms = st.text_input("Enter the message")
# 
# #### 1. Preprocessing
# #### Let us copy the function we creted in the jupyter notebook to do preprocessing
# def transform_text(text):
# 
#     text = text.lower()
#     
#     text = nltk.word_tokenize(text)
#     y = []
#     
#     for i in text:
#     
#         if i.isalnum():
#         
#             y.append(i)
#     text = y[:]
#     
#     #data cannot be copied as text = y always be careful
#     
#     y.clear()
#     
#     for i in text:
#     
#         if i not in stopwords.words('english') and i not in string.punctuation:
#         
#             y.append(i)
#             
#     text = y[:]
#     
#     y.clear()
#     
#     for i in text:
#     
#         y.append(ps.stem(i))
#                 
#                 
#     return " ".join(y)
#     
# 
# transform_sms = transform_text( imput_sms)
# 
# transformed_sms = transform_text( imput_sms)
# 
# #### 2. Vectorize
# vector_imput = tfidf,transform([transformed_sms])
# 
# #### 3. Predict
# result = model.predict(vector_imput)[0]
# 
# #### 4. Display
# if result == 1:
# 
#     st.header("This is a Spam SMS")
#     
# else:
# 
#     st.header("This is a not a Spam SMS")
# 

# #### We must install nltk from the terminal and import string
# #### To run the program go to terminal and write 
# stramlit run app.py

# In[ ]:


#### Let us create some files 
#### first we will create setup.sh and procfile it will run streamlit
##### The third file is gitignore file
#### The last file is requirements.txt and write streamlit, nltk and sklearn in it.
#### To get requirements.txt file run this

