import os
import sys

import spacy
import pandas as pd

from pathlib import Path

def parse_text(text, spacy_model, ego=False):
    '''
    Parses the text using a given spacy model. Additionally has an option to
    prepend "I" to the text for egocentric datasets which helps with parsing as
    these captions tend not to have a subject.
    args:
        text (string):                       the text which will be parsed.
        spacy_model (spacy.lang.en.English): loaded spacy model which will be
                                             used to parse the text.
        ego (bool):                          boolean whether or not to prepend
                                             I to the text for egocentric
                                             datasets.
    returns:
        (spacy.tokens.doc.Doc): The parsed sentence as a spacy Doc.
    '''
    if ego:
        return spacy_model(f'I {text}')
    else:
        return spacy_model(text)

def get_pos(doc, pos):
    '''
    Returns all the words with a given part of speech from the spacy doc.
    args:
        doc (spacy.tokens.doc.Doc): A parsed sentence
        pos (string):               The part of speech to use, must be
                                    uppercase, e.g. VERB, NOUN.
    returns:
        (list of string): List containing the string version of all words which
                          were found to have a specific part of speech.
    '''
    return [w.text for w in doc if w.pos_ == pos]

def get_non_stop(sentence, stop_words):
    '''
    Returns all words which aren't stop words from a given sentence.
    args:
        sentence (string): the sentence to find the non stop words from.
        stop_words(set of string): set of stop words to exclude
    returns
        (list of string): List containing string version of all words which
                          aren't stop words.
    '''
    return [w for w in sentence.split(' ') if w not in stop_words] 

def load_spacy_model(spacy_model_name):
    '''
    loads a spacy model with a given name.
    args:
        spacy_model_name (string): name of the spacy model to load.
    returns:
        (spacy.lang.en.English): loaded spacy model which will be used to parse
                                 the text.
    '''
    try:
        model = spacy.load(spacy_model_name)
    except:
        RaiseException(f"Spacy model {spacy_model_name} not found, maybe it isn't downloaded?")
    return model

def parse_dataframe(df, spacy_model):
    '''
    parses narrations from a dataframe and returns a new version with verbs,
    nouns and stop words as separate columns.
    args:
        df (pandas.DataFrame):               pandas dataframe containing a
                                             narration column which will be parsed.
        spacy_model (spacy.lang.en.English): loaded spacy model which will be used to parse
                                 the text.
    returns:
        (pandas.DataFrame): new pandas dataframe containing the narrations
                            column as well as three new columns (parsed_verbs,
                            parsed_nouns and non_stop_words).
        
    '''
    stop_words = spacy_model.Defaults.stop_words
    parsed_series = df.narration.apply(lambda x: parse_text(x, spacy_model))
    new_df = df.copy()
    new_df['parsed_verbs'] = parsed_series.apply(lambda x: get_pos(x, 'VERB'))
    new_df['parsed_nouns'] = parsed_series.apply(lambda x: get_pos(x, 'NOUN'))
    new_df['non_stop_words'] = df.narration.apply(lambda x: get_non_stop(x, stop_words))
    return new_df

def main(args):
    df = pd.read_pickle(args.INPUT_DF)
    spacy_model = load_spacy_model(args.spacy_model)
    out_df = parse_dataframe(df, spacy_model)
    out_df.to_pickle(args.OUTPUT_DF)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Parser for datafranes')
    parser.add_argument('INPUT_DF', type=Path, help='Path of input DF')
    parser.add_argument('OUTPUT_DF', type=Path, help='Path of output DF')
    parser.add_argument('--spacy-model', type=str, help='Spacy model to use for parsin')

    parser.set_defaults(
        spacy_model='en_core_web_lg'
    )

    main(parser.parse_args())
