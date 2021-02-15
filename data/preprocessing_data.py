import pandas as pd
import zipfile
import requests
from io import BytesIO
import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string

def create_dataset_sent140(with_preprocessing=False):
    """Create a working dataset based on the SENT140 dataset.

    :return a pd.DataFrame object, a dataset with 'id', 'date', 'user',
    'tweet' columns, ordered by 'date'
    """

    # problem 1: read csv-s imbedded into a zip folder - solution:
    # https://stackoverflow.com/questions/40009022/extract-files-inside-zip-sub-folders-with-python-zipfile

    # problem 2: encoding problem - figuring out dataset is actually in latin-1
    # encoding - help: https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte

    # problem 3: date formatting, what to do w tzinfo?
    # solution: tz is ignored

    def format_datetime(s):
        """format 'date' column within dataset

        :return a datetime.datetime object that looks like this:
        2009-04-06 22:19:45
        """
        s = s.split()
        # original data looks like this: 'Mon Apr 06 22:19:45 PDT 2009'

        # tzinfo can be env specific - for our task, it is not
        # relevant --> I am removing it here
        words = s[:4] + s[5:]
        s = ' '.join(words)
        dt = datetime.datetime.strptime(s, '%a %b %d %H:%M:%S %Y')
        return dt

    COLUMNS = ['polarity', 'id', 'date', 'query', 'user', 'tweet']

    url = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(BytesIO(r.content))
    train_df = pd.DataFrame()
    train_df = pd.read_csv(z.open('training.1600000.processed.noemoticon.csv'),
                     encoding="latin-1", names=COLUMNS)

    test_df = pd.DataFrame()
    test_df = pd.read_csv(z.open('testdata.manual.2009.06.14.csv'),
                     encoding="latin-1", names=COLUMNS)


    df = pd.DataFrame()
    df = pd.concat([train_df, test_df])
    df = df.drop(columns=['polarity', 'query'])

    print(f"Number of examples in database: {len(df)}")

    # df kept the indeces of train_df and test_df -> there are duplicate
    # indexing.
    df = df.reset_index(drop=True)

    # Working with date column
    df.date = df.date.apply(format_datetime)
    df.sort_values(by=['date'])

    # Preprocess tweets:
    if with_preprocessing == True:
        df['preprocessed_tweet'] = df.tweet.apply(preprocessing)

    return df

def preprocessing(tweet, level=3):
    res = None

    if level > 0:       # basic tweet syntax removal
        # remove hashtag signs
        tweet = re.sub(r'#', '', tweet)

        # remove "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)

        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

        res = tweet

    if level > 1:       # tokenizing, removing stopwords and punctuation

        # instantiate tokenizer class
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                   reduce_len=True)

        # tokenize tweets
        tweet_tokens = tokenizer.tokenize(tweet)

        # Import the english stopwords
        stopwords_english = stopwords.words('english')

        tweets_clean = []

        for word in tweet_tokens:  # Go through every word in your tokens list
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation):  # remove punctuation
                tweets_clean.append(word)

        res = tweets_clean

    if level > 2:       # stemming
        # Instantiate stemming class
        stemmer = PorterStemmer()

        # Create an empty list to store the stems
        tweets_stemmed = []

        for word in tweets_clean:
            stem_word = stemmer.stem(word)  # stemming word
            tweets_stemmed.append(stem_word)  # append to the list

        res = tweets_stemmed

    return res

if __name__ == "__main__":
    create_dataset_sent140()