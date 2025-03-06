from warnings import filterwarnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud

filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

df = pd.read_excel("amazon.xlsx")  # Reading the Excel file
df.head()
df.info()

df["Review"] = df["Review"].str.lower()  # text to lower case
df["Review"] = df["Review"].str.replace('[^\w\s]', '', regex=True)  # removing punctuations
df["Review"] = df["Review"].str.replace('\d', '', regex=True)  # removing numbers from text

# nltk.download("stopwords") using nltk stopwords library to eliminate stopwords from text
from nltk.corpus import stopwords

sw = stopwords.words("english")
df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


# print(temp_df.head())

drops = pd.Series(" ".join(df["Review"]).split()).value_counts()[-1000:]


df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))  # drop rarewords

# nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
df["Review"] = df["Review"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

# print(df["Review"])

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
# creating a new data frame tf and finding word frequencies in the dataframe

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

var = tf[tf["tf"] > 500]  # take the ones which their frequency is greater than 500
# -------------BAR PLOT--------------
var.plot.bar(x="words", y="tf")  # bar plot visualization
# plt.show()

# -------------WORD CLOUD-------------
text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# ----------SENTIMENT ANALYSIS----------

# nltk.download("vader_lexicon")  # using vader lexicon model for sentiment analysis
sia = SentimentIntensityAnalyzer()
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))
df["polarity_scores"] = df["Review"][0:10].apply(
    lambda x: sia.polarity_scores(x)["compound"])  # find the polarity scores for first 10 observations

df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
print(df["sentiment_label"])

df.groupby("sentiment_label")["Star"].mean()

# -------------- MACHINE LEARNING------------------

train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["sentiment_label"],
                                                    random_state=42)

# I need to turn these text value to numerical values to handle independent variable

# fitting using tfidf vectorizer
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)

x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# ----------LOGISTIC REGRESSION----------
log_model = LogisticRegression()
log_model.fit(x_train_tf_idf_word, train_y)

y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

random_review = pd.Series(df["Review"].sample(1).values)
new_review = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(new_review)

print(f'Review: {random_review[0]} \n Prediction: {pred}')

# ----------RANDOM FOREST----------
rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()
