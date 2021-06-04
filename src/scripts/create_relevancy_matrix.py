import os
import sys

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from nltk.translate import meteor_score as ms


def get_col_name(proxy_type):
    '''
    returns the column name(s) in the dataframe which refers to the type of
    proxy measure being used.
    args:
        proxy_type (string): string representing a proxy measure to split on.
    returns:
        (list of strings) list of columns.
    '''
    if proxy_type == 'BoW':
        col_name = ['non_stop_words']
    elif proxy_type == 'PoS':
        col_name = ['parsed_verbs', 'parsed_nouns']
    elif proxy_type == 'SYN':
        col_name = ['parsed_verb_classes', 'parsed_noun_classes']
    else:
        col_name = ['narration']
    return col_name


def get_meteor_score(sent_1, sent_2):
    '''
    returns the METEOR score between two sentences
    args:
        sent_1 (string): first sentence (used as reference).
        sent_2 (string): second sentence.
    returns:
        (float) METEOR score, value between 0 and 1.
    '''
    return ms.single_meteor_score(sent_1, sent_2)


def create_iou(set_1, set_2, weight=1.0):
    '''
    finds the intersection over union between two sets. Optionally applies a
    weight to the IoU.
    args:
        set_1 (set of items): set of items from the first caption.
        set_2 (set of items): set of items from the second caption.
        weight (float):       optional parameter which is used to weight the
                              resulting IoU score
    returns:
        (float) IoU score, unweighted this value lies between 0 and 1.
    '''
    intersect_ = set_1.intersection(set_2)
    union_ = set_1.union(set_2)
    denominator = len(union_) if len(union_) != 0 else 0.0001
    return weight * len(intersect_) / denominator


def create_relevancy_matrix(df, cols, df2=None, is_meteor=False):
    '''
    Creates a relevancy matrix of size NxM where N is the number of videos and
    M is the number of captions using one of the proxy measures.
    args:
        df (pandas.DataFrame):  dataframe including the list of videos and
                                corresponding captions.
        cols (list of string):  list of column names. A separate relevancy
                                matrix is created for each column and averaged.
        df2 (pandas.DataFrame): dataframe including the list of captions.
                                Optional and only used if the number of
                                captions in the dataset is different from the
                                number of videos.
        is_meteor (bool)      : boolean flag for whether the METEOR scoring
                                function is used or the IoU function is used
    returns:
        (numpy.Array): Array of size NxM where the ith/jth element refers to
                       the ground truth relevancy between the ith video and the
                       jth caption. Note that for METEOR this matrix is not
                       symmetric.
    '''
    if df2 is None:
        mat_shape = (len(df), len(df))
    else:
        mat_shape = (len(df), len(df2))
    rel_mat = np.zeros(mat_shape)
    for col in cols:
        if col not in df.columns:
            raise Exception(f'Column: {col} not found in dataframe')
        vals1 = df[col].values
        if df2 is None:
            vals2 = df[col].values
            mat_shape = (len(df), len(df))
        else:
            vals2 = df2[col].values
            mat_shape = (len(df), len(df2))

        for i, v1 in enumerate(tqdm(vals1)):
            for j, v2 in enumerate(vals2):
                if is_meteor:
                    rel_mat[i][j] += get_meteor_score(v1, v2)
                else:
                    rel_mat[i][j] += create_iou(set(v1), set(v2))
    return rel_mat / len(cols)


def main(args):
    input_df = pd.read_pickle(args.INPUT_DF)
    if args.second_df is not None:
        input_df2 = pd.read_pickle(args.second_df)
    else:
        input_df2 = None
    col_name = get_col_name(args.PROXY_TYPE)
    rel_mat = create_relevancy_matrix(input_df, col_name, input_df2, is_meteor=args.PROXY_TYPE=='MET')
    pd.to_pickle(rel_mat, args.OUTPUT_PKL)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Create Relevancy Matrix from parsed dataframe')
    parser.add_argument('INPUT_DF', type=Path, help='Path of input DF')
    parser.add_argument('OUTPUT_PKL', type=Path, help='Path of output pickle')
    parser.add_argument('PROXY_TYPE', type=str, choices=['BoW', 'PoS', 'SYN', 'MET'], help='Which proxy type to create')
    parser.add_argument('--second-df', type=Path, help='Path of opposing modality if different')

    main(parser.parse_args())
