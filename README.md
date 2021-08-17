# Semantic-Video-Retrieval
This repo contains code to evaluate for the semantic similarity video retrieval task, including:
* An example to generate a pandas dataframe from json annotations for YouCook2.
* A script to parse the captions using spacy.
* An optional script to create synset information using WordNet features.
* A script to create the ground truth relevancy matrix from the four proxy measures listed in the paper: Bag of Words, Part of Speech, Synset, METEOR.

More information about Semantic Similarity for Video Retrieval can be found on the [webpage](https://mwray.github.io/SSVR).

## Setup

Python 3.8 was used with the following libraries:
* argparse
* ast
* nltk
* numpy
* pandas
* pathlib
* spacy
* tqdm
* json (optional for the scripts but useful if reading in json files)

## Quick Start Guide

1. Construct a pandas dataframe of the captions from a tran/val/test split (see below for an example).
2. Run `parse_dataframe` in order to use spacy to parse the captions. This can be run with the command `python -m scripts.parse_dataframe INPUT_DF OUTPUT_DF` (Additionally, the spacy model can be parsed in with the optional `--spacy-model` argument)
3. (Optional) Run `create_synsets` to generate synset information on the dataframe. This is only required for the synset proxy measure (as it can take a while!). E.g. `python -m scripts.create_synsets INPUT_DF OUTPUT_DF
4. Run `create_relevancy_matrix` to generate the ground truth similarity matrix for the pairs of captions. For example: `python -m scripts.create_relevancy_matrix INPUT_DF OUTPUT_DF BoW` will create a matrix using the Bag of Words (BoW) proxy metric.

Code to evaluate using the Normalised Discounted Cumulative Gain Metric (nDCG) can be found [here](https://github.com/mwray/Joint-Part-of-Speech-Embeddings/blob/main/src/evaluation/nDCG.py).
This requires the relevancy matrix created above and a similarity matrix which can be found via a dot product of l2 normalised video and text features.
For example, if the dataset has N videos and M captions and you're using a model with an embedding space size of D, then you must first extract and normalise the video features, V, (size NxD) and text features, T, (size MxD), then the similarity matrix is given by ``V.dot(T.transpose())`.

## Other Considerations
* When creating synsets, if these are known beforehand `--verb-classes` and `--noun-classes` exist as optional parameters to pass in csv files containing synset information (see [EPIC-KITCHENS-100 Verb CSV](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/EPIC_100_verb_classes.csv) for an example of what this looks like).
* If the size of each modality is different, then `create_relevancy_matrix` can be passed the corresponding dataframe for the second modality using the `--second-df` parameter.


## YouCook2 Example
An example notebook shows the creation of the train/val dataframes necessary for the scripts for YouCook2 in `./notebooks/YouCook2_example`. This represents step 1 of the Quick Start Guide above.

## Results

Here we show the up-to-date results for 3 datasets using the Semantic Similarity Video Retrieval Task.: YouCook2[1], MSR-VTT[2] and EPIC-Kitchens-100[3].
Results are given in nDCG, averaged across video-to-text and text-to-video retrieval.
*denotes results trained with a simple MLP baseline. See the paper for more information on this baseline.

### YouCook2

|         |   BoW  |   PoS  |   Syn  |   MET  |
|---------|--------|--------|--------|--------|
| Random  |  23.1  |  22.1  |  27.7  |  66.2  |
| MEE*    |**42.1**|**40.3**|**45.3**|**73.3**|
| MoEE[4] |  41.5  |  39.1  |  44.0  |**73.0**|
| CE[5]   |**41.8**|  39.3  |  44.1  |**73.0**|

### MSR-VTT

|         |   BoW  |   PoS  |   Syn  |   MET  |
|---------|--------|--------|--------|--------|
| Random  |  34.0  |  30.0  |  11.6  |  80.4  |
| MEE*    |  51.6  |  48.5  |  33.5  |  83.3  |
| MoEE[4] |**53.9**|**50.8**|**36.8**|**83.9**|
| CE[5]   |**54.0**|**50.9**|**36.7**|***4.0**|

### EPIC-Kitchens-100

|         |   BoW  |   PoS  |   Syn  |   MET  |
|---------|--------|--------|--------|--------|
| Random  |  11.7  |  4.5   |  10.7  |  13.0  |
| MEE*    |**39.3**|  29.2  |  41.8  |  41.0  |
| JPoSE[6]|**39.5**|**30.2**|**49.0**|**44.5**|

## Citation
If you use the code within this repository and/or evaluate for semantic similarity video retrieval please kindly cite:

```
@inproceedings{wray2021semantic,
  title={On Semantic Similarity in Video Retrieval},
  author={Wray, Michael and Doughty, Hazel and Damen, Dima},
  booktitle={CVPR},
  year={2021}
}
```

## Sources
[1] Luowei Zhou, Chenliang Xu, and Jason J Corso. Towards automatic learning of procedures from web instructional videos. CoRR, abs/1703.09788, 2017.

[2] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016.

[3] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Jian Ma, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision. CoRR, abs/2006.13256, 2020.

[4] Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. Use what you have: Video retrieval using representations from collaborative experts. In BMVC, 2019.

[5] Antoine Miech, Ivan Laptev, and Josef Sivic. Learning a text-video embedding from incomplete and heterogeneous data. CoRR, abs/1804.02516, 2018.

[6] Michael Wray, Diane Larlus, Gabriela Csurka, and Dima Damen. Fine-grained action retrieval through multiple parts-of-speech embeddings. In ICCV, 2019.
