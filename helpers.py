import pandas as pd
import re

def clean_text(text):
    url = re.compile(r"\S*https?:\S*")
    special_ch = re.compile('[-\+!~@#$%^_&*()={}\[\]:;<.>?/\'"]')
    new_lines = re.compile('\s+')
    numbers = re.compile('[0-9]')
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)    
    
    text = url.sub("", text)
    text = special_ch.sub("", text)
    text = new_lines.sub(" ", text)
    text = numbers.sub("", text)
    text = emoji_pattern.sub("", text)

    return text.strip()

def check_duplicates(df):
    return bool(df.duplicated().sum())

def check_empty(df):
    return df.isnull().values.any() or df.isna().values.any() 