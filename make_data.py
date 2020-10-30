import string
import pandas as pd
import csv
from nltk import sent_tokenize, word_tokenize
table = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation}

from utils.utils import clean_string


def make_data(train_file_path, test_file_path, text_name, label_name, columns_name = ['Id','Text','Label']):

    df_train = pd.read_csv(train_file_path, sep="\t",  quoting=csv.QUOTE_NONE)
    df_test = pd.read_csv(test_file_path,sep ="\t",  quoting=csv.QUOTE_NONE, names=columns_name)

    df_train[text_name] = df_train[text_name].apply(lambda s : clean_string(s))
    df_test[text_name] = df_test[text_name].apply(lambda s : clean_string(s))

    train_text = df_train[text_name].values.tolist()
    train_label = df_train[label_name].values.tolist()

    test_text = df_test[text_name].values.tolist()
    test_label = df_test[label_name].values.tolist()

    return train_text, train_label, test_text, test_label