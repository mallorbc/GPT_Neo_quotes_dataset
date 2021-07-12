import json
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())


def create_sentences(sentences,key,dict):
    data = dict[key]
    counter = 0
    for item in data:
        #cleans data with these weird entries
        if item.find("[10w]")==-1:
            #skips non ascii
            if not isascii(item):
                continue
            #saves the quotes with style of: <|endoftext|>love: quote about love here<|endoftext|>
            sentences.append("<|endoftext|>" + key + ": " + item + "<|endoftext|>")
    return sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tool to train a custom GPT3 model")
    parser.add_argument("-f", "--file",type=str,default="./quotes.json")
    args = parser.parse_args()
    json_file = args.file
    json_file = os.path.realpath(json_file)
    #loads the file
    with open(json_file) as f:
        data = json.load(f)
    #creates dictonary that will hold categories
    category_dict = {}
    #loops through all quotes and adds them to the appropriate categories
    for index, item in enumerate(data):
        if item["Category"] in category_dict:
            category_dict[item["Category"]].append(item["Quote"])
        else:
            entry = [item["Quote"]]
            category_dict[item["Category"]] = entry
    #remove empty entry
    category_dict.pop("",None)
    #prints out the the different quote categories
    print(category_dict.keys())

    sentences = []
    for key in category_dict.keys():
        #creates the data entries for training the model
       sentences = create_sentences(sentences,key,category_dict)
    #splits into train and test data for training
    train_sentences,test_sentences = train_test_split(sentences,test_size=0.2)
    #creates dataframes that will eventually be saved with "text" column as a csv file
    train_df = pd.DataFrame(train_sentences,columns=['text'])
    validate_df = pd.DataFrame(test_sentences,columns=['text'])
    #drops na items if there are any
    train_df = train_df.dropna()
    validate_df = validate_df.dropna()
    train_df.to_csv("train.csv")
    validate_df.to_csv("validation.csv")


