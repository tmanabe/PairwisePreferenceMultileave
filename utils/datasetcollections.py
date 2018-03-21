# -*- coding: utf-8 -*-

import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset import DataSet


def get_datasets(sim_args):
    """
    Function for retrieving datasets from simulation arguments.
    """
    if len(sim_args.data_folders) == 1 and sim_args.data_folders[0] == 'all':
        data_tags = [
            'Webscope_C14_Set1',
            'Webscope_C14_Set2',
            'MSLR-WEB10k',
            'NP2003',
            'NP2004',
            'HP2003',
            'HP2004',
            'TD2003',
            'TD2004',
            'MQ2007',
            'MQ2008',
            'OHSUMED',
            ]
    elif len(sim_args.data_folders) == 1 and sim_args.data_folders[0] == 'CIKM2017':
        data_tags = [
            'MSLR-WEB10k',
            'NP2003',
            'NP2004',
            'HP2003',
            'HP2004',
            'TD2003',
            'TD2004',
            'MQ2007',
            'MQ2008',
            'OHSUMED',
            ]
    elif len(sim_args.data_folders) == 1 and sim_args.data_folders[0] == 'letor64':
        data_tags = [
            'NP2003',
            'NP2004',
            'HP2003',
            'HP2004',
            'TD2003',
            'TD2004',
            ]
        # random.shuffle(data_tags)
    else:
        data_tags = sim_args.data_folders
    for data_tag in data_tags:
        assert data_tag in DATASET_COLLECTION, 'Command line input is currently not supported.'
        yield DATASET_COLLECTION[data_tag]


PREFIX = '/media/sdb1/letor'
DATASET_COLLECTION = {}
DATASET_COLLECTION['NP2003'] = DataSet('2003_np', PREFIX + '/Gov/Feature_min/2003_np_dataset/Fold*/',
                                       'bin', True, 64,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['NP2004'] = DataSet('2004_np', PREFIX + '/Gov/Feature_min/2004_np_dataset/Fold*/',
                                       'bin', True, 64,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS #19 total
                                             ])
DATASET_COLLECTION['HP2003'] = DataSet('2003_hp', PREFIX + '/Gov/Feature_min/2003_hp_dataset/Fold*/',
                                       'bin', True, 64,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['HP2004'] = DataSet('2004_hp', PREFIX + '/Gov/Feature_min/2004_hp_dataset/Fold*/',
                                       'bin', True, 64,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['TD2003'] = DataSet('2003_td', PREFIX + '/Gov/Feature_min/2003_td_dataset/Fold*/',
                                       'bin', True, 64,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['TD2004'] = DataSet('2004_td', PREFIX + '/Gov/Feature_min/2004_td_dataset/Fold*/',
                                       'bin', True, 64,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])

DATASET_COLLECTION['MQ2008'] = DataSet('MQ2008', PREFIX + '/MQ2008/Fold*/',
                                       'short', True, 46,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41)  #LMIR #25 total
                                             ])
DATASET_COLLECTION['MQ2007'] = DataSet('MQ2007', PREFIX + '/MQ2007/Fold*/',
                                       'short', True, 46,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41)  #LMIR
                                             ])
DATASET_COLLECTION['OHSUMED'] = DataSet('OHSUMED', PREFIX + '/OHSUMED/Feature-min/Fold*/',
                                       'short', True, 45,
                                       multileave_feat=[
                                               #[6,7],       #HITS
                                               range(9,13), #TF-IDF
                                               [28],        #sitemap
                                               range(15,28), #BM25 and LMIR
                                             ])

DATASET_COLLECTION['MSLR-WEB10k'] = DataSet('MSLR-WEB10k', PREFIX + '/MSLR-WEB10K/Fold*/',
                                            'long', False, 136,
                                            multileave_feat=[
                                               range(71,91), #TF-IDF
                                               range(106,111), #BM25
                                               range(111,126), #LMIR # 40 total
                                               # range(96,106), #Boolean Model, Vector Space Model
                                             ])
