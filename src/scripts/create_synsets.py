import os
import sys

import pandas as pd

from ast import literal_eval
from collections import defaultdict
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from tqdm import tqdm

from nltk.corpus import wordnet as wn

def write_class_csv(class_to_word_dict, out_path):
    '''
    writes the class_to_word_dictionary as a csv in the same way as
    https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/EPIC_100_verb_classes.csv
    args:
        class_to_word_dict (dict of string -> list of strings): dictionary
                containing the classes/synsets as keys and the values as a list
                of words as the members or each class/synset
        out_path (pathlib.Path): Path to csv file to save contents of
                                 dictionary to.
    '''
    with open(out_path, 'w') as out_f:
        out_f.write('id,key,instances\n')
        for class_ in class_to_word_dict:
            words = class_to_word_dict[class_]
            out_f.write(f'{class_},{words[0]},"{str(words)}"\n')

def save_reverse_class_dict(word_to_class_dict, original_path):
    '''
    reverses the word to class dictonary and saves it as a csv.
    args:
        word_to_class_dict (dict of string -> string): dictionary containing
                words as keys and the values their corresponding class/synset.
        out_path (pathlib.Path): Path to csv file to save contents of
                                 the reversed dictionary to.
    '''
    class_to_word_dict = {}
    for word in word_to_class_dict:
        class_ = word_to_class_dict[word]
        if class_ not in class_to_word_dict:
            class_to_word_dict[class_] = [word]
        else:
            class_to_word_dict[class_].append(word)
    write_class_csv(class_to_word_dict, original_path.parent / f'mod_{original_path.parts[-1]}')


def read_classes_df(in_csv):
    '''
    reads in the input csv file containing the class/synset information into a
    pandas dataframe.
    args:
        in_csv (pathlib.Path): Path to CSV containing "id, key, instances"
    returns:
        (pandas.DataFrame): dataframe format of the input csv.
    '''
    return pd.read_csv(in_csv, index_col="id", converters={"instances": literal_eval})

def read_classes(in_csv):
    '''
    reads in the input csv containing class/synset information into a dictionary
    args:
        in_csv (pathlib.Path): Path to CSV containing "id, key, instances"
    returns:
        (dict string -> list of string)
    '''
    result = read_classes_df(in_csv)

    classes = {}
    for cls in result.index:
        classes[cls] = result.loc[cls]["instances"]
    return classes

def reverse_class_dict(classes):
    '''
    Reverses a dictionary containing class/synset information, keys become
    values and vice versa
    args:
        classes (dict string -> string): dictionary containing class/synset
                                         information
    returns:
        (dict string -> string): reversed dictionary.
    '''
    seen_elements = set()
    duplicates = defaultdict(lambda: [])
    for key, elements in classes.items():
        for element in elements:
            if element in seen_elements:
                duplicates[element].append(key)
            else:
                seen_elements.add(element)
    if len(duplicates) > 0:
        raise ValueError(
            "Values were present across multiple keys\n" + str(dict(duplicates))
        )

    return {v: k for k in classes for v in classes[k]}


def get_wn_synset(words, search_word, pos, lemmatiser):
    '''
    returns a WordNet synset of a search word based on the context of a
    sentence and a specific part of speech. If the synset cannot be found then
    the search word will be returned as a single size synset.
    args:
        words (list of string): words of the sentence to be used as context.
                                Note this is split using the split function.
        search_word (string):   search word to find the synset for.
        pos (string):           part of speech to search for, note this is an
                                attribute from nltk.corpus and can be wn.VERB
                                ('v'), wn.NOUN ('n'), wn.ADJ ('a')
    returns:
        ([nltk.corpus.reader.wordnet.Synset, string]): wordnet synset of the
                    search word, or if the synset cannot be found, the
                    search_word input arg.
    '''
    search_word_lemma = lemmatiser.lemmatize(search_word, pos)
    synset = lesk(words, search_word_lemma, pos=pos)
    if synset is not None:
        synset = synset.name()
    else:
        synset = search_word
    return synset


def get_dict_synset(search_word, word_to_syn_dict, pos, lemmatiser):
    '''
    returns a synset of a word from the synset/class dictionary. If the synset
    cannot be found within the class/synset dictionary, then a new synset is
    added to the dictionary.
    args:
        search_word (string):   search word to find the synset for.
        word_to_syn_dict (dict string -> [string, int]): dictionary. Keys
                represent the words and values the class/synset that word
                belongs to. In the case of a word that wasn't found, then a new
                entry will be added with a numerical name.
    returns:
        Tuple(
            [string, int] synset of search word,
            dict string -> [string, int] Updated word_to_syn_dict
        )
    '''
    search_word_lemma = lemmatiser.lemmatize(search_word, pos)
    if search_word_lemma not in word_to_syn_dict:
        word_to_syn_dict[search_word_lemma] = len(word_to_syn_dict)
    return word_to_syn_dict[search_word_lemma], word_to_syn_dict


def create_synsets(df, narration_col, word_col, pos, word_to_syn_dict=None):
    '''
    Creates Synsets series from information within a dataframe column either by
    using WordNet synset information (default) or by using external
    synset/class information from an optional dictionary parameter.
    args:
        df (pandas.DataFrame):  dataframe to use to generate synset column and
                                to modify.
        narration_col (string): name of the column which refers to the sentence
                                used as context for the lesk algorithm when
                                finding the WordNet synsets.
        word_col (string):      name of the column which will contain words to
                                find synsets for.
        pos (string):           part of speech to search for, note this is an
                                attribute from nltk.corpus and can be wn.VERB
                                ('v'), wn.NOUN ('n'), wn.ADJ ('a')
        word_to_syn_dict (dict string -> [string, int]): dictionary. Keys
                represent the words and values the class/synset that word
                belongs to. This is an optional parameter with the default
                operation of this function to grab synsets information from
                WordNet. Note that this dictionary will be updated in the case
                of missing words.
    returns:
        Tuple(
            pandas.Series: Updated DataFrame with synset column added.
            dict string -> [string, int] Updated word_to_syn_dict
        )
    '''
    synsets_dict = {}
    sentence_word_tuples = zip(df[narration_col].values, df[word_col].values)
    indices = list(df.index)
    lemmatiser = WordNetLemmatizer()
    for i, (sentence, search_words) in enumerate(sentence_word_tuples):
        words = sentence.split(' ')
        synset_set = set()
        for search_word in search_words:
            if word_to_syn_dict is None:
                synset = get_wn_synset(words, search_word, pos, lemmatiser)
            else:
                synset, word_to_syn_dict = get_dict_synset(search_word, word_to_syn_dict, pos, lemmatiser)
            synset_set.add(synset)
        synsets_dict[indices[i]] = list(synset_set)
    return pd.Series(synsets_dict), word_to_syn_dict


def main(args):
    input_df = pd.read_pickle(args.INPUT_DF)
    output_df = input_df.copy()
    verb_class_dict = noun_class_dict = None
    if args.verb_classes:
        verb_class_dict = reverse_class_dict(read_classes(args.verb_classes))
    if args.noun_classes:
        noun_class_dict = reverse_class_dict(read_classes(args.noun_classes))
    verbs, verb_class_dict = create_synsets(input_df, 
                                            'narration', 
                                            'parsed_verbs', 
                                            wn.VERB, 
                                            verb_class_dict)
    nouns, noun_class_dict = create_synsets(input_df, 
                                            'narration', 
                                            'parsed_nouns', 
                                            wn.NOUN, 
                                            noun_class_dict)
    output_df['parsed_verb_classes'] = verbs
    output_df['parsed_noun_classes'] = nouns
    output_df.to_pickle(args.OUTPUT_DF)
    #If csv class/synset information was used, save the updated versions!
    if args.verb_classes:
        save_reverse_class_dict(verb_class_dict, args.verb_classes)
    if args.noun_classes:
        save_reverse_class_dict(noun_class_dict, args.noun_classes)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Find synsets of words in dataframe')
    parser.add_argument('INPUT_DF', type=Path, help='Path of input DF')
    parser.add_argument('OUTPUT_DF', type=Path, help='Path of output pickle')
    parser.add_argument('--verb-classes', type=Path, help='Path to synset info for verbs')
    parser.add_argument('--noun-classes', type=Path, help='Path to synset info for nouns')

    main(parser.parse_args())
