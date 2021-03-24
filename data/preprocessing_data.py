import pandas as pd
import numpy as np
import zipfile
import requests
from io import BytesIO
import datetime
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string

def write_to_csv(df, outdir, filename):
    """Save pd.DataFrame to .csv file

    Parameters:
        outdir (str): relative path to folder
        filename (str): filename ending with .csv
    Returns:
        -
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, filename)

    df.to_csv(fullname)

def preprocessing(tweet, level=2):
    """Preprocess tweets

    Parameters:
        tweet (str): tweet text to be modified
        level (int): level of preprocessing. Possible values:
            1: basic tweet syntax removal
            2: basic tweet syntax removal + tokenizing and dealing with punctuation
            3: basic tweet syntax removal + tokenizing and dealing with punctuation + stemming
    Returns:
        list(str): a list of words from the preprocessed tweets
    """
    res = None

    if level > 0:  # basic tweet syntax removal
        # remove hashtag signs
        tweet = re.sub(r'#', '', tweet)

        # remove "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)

        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

        # note end of sentences: <EOS>
        tweet = re.sub(r'[.?!]+', ' <EOS>', tweet)

        # remove mentions
        tweet = re.sub(r'^@[\s]+', '', tweet)

        res = tweet

    if level > 1:  # tokenizing and dealing with punctuation

        # instantiate tokenizer class
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                   reduce_len=True)

        # tokenize tweets
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []

        if level == 2:
            for word in tweet_tokens:  # Go through every word in your tokens list
                if (word not in string.punctuation):  # remove punctuation
                    tweets_clean.append(word)

        res = tweets_clean

    if level > 2:  # stemming
        nltk.download('stopwords')

        # Import the english stopwords
        stopwords_english = stopwords.words('english')
        tweets_clean = []

        for word in tweet_tokens:  # Go through every word in your tokens list
            if (word not in string.punctuation and word not in stopwords_english):  # remove punctuation & stopwords
                tweets_clean.append(word)

        # Instantiate stemming class
        stemmer = PorterStemmer()

        # Create an empty list to store the stems
        tweets_stemmed = []

        for word in tweets_clean:
            stem_word = stemmer.stem(word)  # stemming word
            tweets_stemmed.append(stem_word)  # append to the list

        res = tweets_stemmed

    return res

def create_dataset_sent140(with_preprocessing=True):
    """Create a working dataset based on the SENT140 dataset.

    Arguments:
        with_preprocessing: if True, this function invokes preprocessing on 'tweet' column
    Returns:
         a pd.DataFrame object, a dataset with 'id', 'date', 'user', 'tweet'[, 'processed_tweet'] columns, ordered by 'date'
    """

    # problem 1: read csv-s embedded into a zip folder - solution:
    # https://stackoverflow.com/questions/40009022/extract-files-inside-zip-sub-folders-with-python-zipfile

    # problem 2: encoding problem - figuring out dataset is actually in latin-1
    # encoding - help: https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte

    # problem 3: date formatting, what to do w tzinfo?
    # solution: tz is ignored

    def format_datetime(s):
        """format 'date' column within dataset

        Arguments:
            s (str): string to be formatted to datetime.datetime
        Returns:
             a datetime.datetime object that looks like this: 2009-04-06 22:19:45
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

    print("\nDownloading Sentiment 140 dataset from the internet...\n")
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
    print("\nOrganising data...\n")
    # df kept the indeces of train_df and test_df -> there are duplicate
    # indexing.
    df = df.reset_index(drop=True)

    # Working with date column
    df.date = df.date.apply(format_datetime)
    df.sort_values(by=['date'])

    df.index.rename('ind', inplace=True)

    # Preprocess tweets:
    print("\nPreprocessing tweets...\n")
    if with_preprocessing:
        df['preprocessed_tweet'] = df.tweet.apply(preprocessing, level=2)

    outdir = './csv'
    filename = '01-preprocessed.csv'
    write_to_csv(df, outdir, filename)

    return df

def rename_users(df):
    """Rename users in a certain logic: Monica, Phoebe, Rachel, Chandler, Joey, Ross. It also saved to result in a csv.

    Returns:
         a pd.DataFrame object, with a new user column: 'renamed'.
    """

    def rename_user(ind):
        """Gets an index and returns a new name for the user.

        Arguments:
            ind (int): index of the row
        Returns:
            (str): the new name
        """
        if ind % 2 == 0:
            if ind % 5 == 0:
                return 'Rachel'
            elif ind % 3 == 0:
                return 'Phoebe'
            else:
                return 'Monica'
        else:
            if ind % 5 == 1:
                return 'Ross'
            elif ind % 3 == 1:
                return 'Joey'
            else:
                return 'Chandler'

    print('\nRenaming users...\n')
    df['ind'] = df.index
    df['renamed_user'] = df.loc[:, 'ind'].apply(rename_user)

    df.set_index('ind', inplace=True)
    outdir = './csv'
    filename = '02-preprocessed_renamed.csv'
    write_to_csv(df, outdir, filename)

    return df

