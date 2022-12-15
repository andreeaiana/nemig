from typing import Any, Set, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api

from src.utils.data_utils import check_integrity, load_cache, update_cache
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class EntityFilter():
    def __init__(
            self,
            language: str,
            overwrite_cache: bool,
            threshold_ner_score: float,
            threshold_nel_score_text: float,
            threshold_nel_score_keywords: float,
            author_person_description: List[str],
            author_person_occupation: List[str],
            author_org_description: List[str],
            author_org_instanceof: List[str],
            wiki_prop4ent_type: Dict[str, List[str]],
            columns2filter: List[str],
            text_columns: List[str],
            author_columns: List[str],
            keywords_columns: List[str],
            input_data_files: Dict[str, str],
            results_files: Dict[str, Dict[str, str]],
            qid2attr_filepath: str
            ) -> None:
        
        self.language = language
        self.overwrite_cache = overwrite_cache

        self.threshold_ner_score = threshold_ner_score
        self.threshold_nel_score_text = threshold_nel_score_text
        self.threshold_nel_score_keywords = threshold_nel_score_keywords

        self.author_person_description = author_person_description
        self.author_person_occupation = set(author_person_occupation)
        self.author_org_description = author_org_description
        self.author_org_instanceof = set(author_org_instanceof)
        self.wiki_prop4ent_type = wiki_prop4ent_type

        self.columns2filter = columns2filter
        self.text_columns = text_columns
        self.author_columns = author_columns
        self.keywords_columns = keywords_columns

        self.input_data_files = input_data_files
        self.results_files = results_files
        self.qid2attr_filepath = qid2attr_filepath

        assert all([col in self.input_data_files.keys() for col in columns2filter])

        # initialize wikidata qids to attributes mapping
        if check_integrity(self.qid2attr_filepath):
            log.info(f'Mapping of Wikidata QIDs to attributes already cached at {qid2attr_filepath}. Loading into memory.')
            self.qid2attr = load_cache(self.qid2attr_filepath)
        else:
            log.info(f'The mapping of Wikidata QIDs to attributes not created yet. Initializing empty map.')
            self.qid2attr = dict()

    def filter_entities(self) -> None:
        # Create qid2attr map if not already cached
        if not self.qid2attr:
            self.map_qid2attr()

        # Filter entities
        for col in self.columns2filter:
            
            if not self.overwrite_cache and check_integrity(self.results_files['filtered'][col]):
                log.info(f"Filtered entities for articles' {col} already cached at {self.results_files['filtered'][col]}.")
            else:
                log.info(f"Filtering entities for articles' {col}.")

                # load data
                data = pd.read_csv(self.input_data_files[col])
            
                # copy of unfiltered data
                unfiltered_data = data.copy()

                # text-specific filtering
                if col in self.text_columns:
                    # filter on NER score
                    data = self._filer_on_ner_score(data)
        
                    # filter entities which don't have a Wikidata page anymore
                    data = self._filter_no_wikipage(data)
                
                    # filter entities which point to a Wikimedia disambiguation page
                    data = self._filter_wikimedia_disambiguation_page(data)

                    # filter entities whose entity type does not match the tpe of entity in Wikidata
                    data = self._filter_on_entity_type(data)

                    # filter on named entity linking score
                    data = self._filter_on_nel_score(data, self.threshold_nel_score_text)
                
                    # filter entities with foreign labels that only appear once
                    data = self._filter_on_language(data)

                # author-specific filtering
                if col in self.author_columns:
                    # filter entities recognised as LOC 
                    data = self._filter_incorrect_author_type(data, 'LOC')

                    # filter entities recognised as MISC
                    data = self._filter_incorrect_author_type(data, 'MISC')

                    # filter correctly and incorrectly linked entities
                    data = self._filter_authors(data)

                # keywords-specific filtering
                if col in self.keywords_columns:
                    # drop entities with NaN QID
                    log.info('Filtering entities with NaN Wikidata QIDs.')
                    column_name = col if col=='topic_label' else 'news_keywords'
                    nan_wiki_entry = data[data['wikidata_id'].isnull()].index
                    data = data.drop(nan_wiki_entry)
                    log.info(f'Filtered {len(nan_wiki_entry)} entities.')

                    # filter not extracted entities without a wikidata page
                    log.info('Filtering not extracted entities without a Wikidata page.')
                    entities_no_wikipage = [idx for idx in tqdm(data.index.to_list()) if data.loc[idx]['wikidata_id'] not in self.qid2attr.keys()]
                    entities_not_extracted = data[data['word'].isnull()].index
                    rows2drop = set(entities_no_wikipage).intersection(set(entities_not_extracted))
                    data = data.drop(rows2drop)
                    log.info(f'Filtered {len(rows2drop)} entities.')

                    # sample entities with no wikidata page
                    entities_no_wikipage = [idx for idx in tqdm(data.index.to_list()) if data.loc[idx]['wikidata_id'] not in self.qid2attr.keys()]
                    data_no_wikipage = data.loc[entities_no_wikipage]
                    data_no_wikipage['linked_entity'] = data_no_wikipage[column_name].apply(lambda x: x.strip('[START]').strip('[END]').strip().lower())
                    data_no_wikipage['linked_entity_language'] = data_no_wikipage['linked_entity_language'].apply(lambda _: None)
                    data_no_wikipage['nel_score'] = data_no_wikipage['nel_score'].apply(lambda _: None)
                    data_no_wikipage['wikidata_id'] = data_no_wikipage['wikidata_id'].apply(lambda _: None)

                    data = data.drop(entities_no_wikipage)
                    
                    # filter entities which point to a Wikimedia disambiguation page
                    data = self._filter_wikimedia_disambiguation_page(data)

                    # filter entities whose entity type does not match the tpe of entity in Wikidata
                    data = self._filter_on_entity_type(data)

                    # filter on named entity linking score
                    incorrectly_linked = data[data['nel_score'] < self.threshold_nel_score_keywords] 
                    incorrectly_linked['linked_entity'] = incorrectly_linked[column_name].apply(lambda x: x.strip('[START]').strip('[END]').strip().lower())
                    incorrectly_linked['linked_entity_language'] = incorrectly_linked['linked_entity_language'].apply(lambda _: None)
                    incorrectly_linked['nel_score'] = incorrectly_linked['nel_score'].apply(lambda _: None)
                    incorrectly_linked['wikidata_id'] = incorrectly_linked['wikidata_id'].apply(lambda _: None)
                    data = self._filter_on_nel_score(data, self.threshold_nel_score_keywords)
                
                    # filter entities with foreign labels that only appear once
                    data = self._filter_on_language(data)

                    # concatenate all results
                    data = pd.concat([data, data_no_wikipage, incorrectly_linked])

                # cache data entries removed during filtering
                removed_data = unfiltered_data.drop(data.index)
                log.info(f'Filtered {len(removed_data)} entities in total.')
                removed_data.to_csv(self.results_files['removed'][col], index=False)
                
                # cache filtered data
                data = data.reset_index(drop=True)
                data.to_csv(self.results_files['filtered'][col], index=False)
    
    def map_qid2attr(self) -> None:
        log.info('Creating mapping of Wikidata QIDs to attributes and surface forms.')
        qids = []

        log.info('Retrieving entities from data files.')
        for file in tqdm(self.input_data_files.values()):
            df = pd.read_csv(file)
            entities = df['wikidata_id'].tolist()
            qids.extend(entities)

        unique_qids = set(qids)
        if np.nan in unique_qids:
            unique_qids.remove(np.nan)
        log.info(f'Retrieved {len(qids)} non-unique entities and {len(unique_qids)} unique entities from {len(self.input_data_files)} data files.')
        
        self.qid2attr = self._map_qid2attr(unique_qids)

    def _map_qid2attr(self, qids: Set[str]) -> Dict[str, Any]:
        log.info('Retrieving information from Wikidata.')
        qid2attr = dict()
        steps = 0
        for qid in tqdm(list(qids)):
            try:
                attr_dict = get_entity_dict_from_api(qid)
                qid2attr[qid] = attr_dict
            except Exception:
                continue
            
            if steps % 5000 == 0:
                update_cache(qid2attr, self.qid2attr_filepath)
            steps += 1

        update_cache(qid2attr, self.qid2attr_filepath)
        log.info(f'Finished query Wikidata. Extracted information for {len(self.qid2attr)} entities.')

        return qid2attr

    def _map_qid2surface_form(self, data: pd.DataFrame) -> Dict[str, Set[Tuple[str, str]]]:
        log.info('Creating mapping of Wikidata QIDs to surface forms.')
        qid2surface_form = defaultdict(set)
        
        for idx in tqdm(range(len(data))):
            qid = data['wikidata_id'].iloc[idx]
            word = data['word'].iloc[idx]
            language = data['linked_entity_language'].iloc[idx]

            if qid != None:
                qid2surface_form[qid].add((word, language))

        return qid2surface_form
         
    def _filer_on_ner_score(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info('Filtering on named entity recognition score.')
        init_data_len = len(data)
       
        data = data[data['ner_score'] >= self.threshold_ner_score]
        log.info(f'Filtered {init_data_len - len(data)} entities.')

        return data

    def _filter_no_wikipage(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info('Filtering entities without a Wikidata page.')
        entities_no_wikipage = []
        for idx in tqdm(data.index.to_list()):
            if data.loc[idx]['wikidata_id'] not in self.qid2attr.keys():
                entities_no_wikipage.append(idx)
        
        data = data.drop(entities_no_wikipage)
        log.info(f'Filtered {len(entities_no_wikipage)} entities.')

        return data

    def _filter_wikimedia_disambiguation_page(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info('Filtering Wikimedia disambiguation pages.')
        wikimedia_disambig_page = []
        for idx in tqdm(data.index.to_list()):
            qid = data.loc[idx]['wikidata_id']
            if self._check_property_values(qid, 'P31', set(['Q4167410'])):
                wikimedia_disambig_page.append(idx)
       
        data = data.drop(wikimedia_disambig_page)
        log.info(f'Filtered {len(wikimedia_disambig_page)} entities.')

        return data

    def _filter_on_entity_type(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info('Filtering entities based on entity type.')
        for ent_type in self.wiki_prop4ent_type.keys():
            log.info(f'Filtering for entity type: {ent_type}.')
            sample = data[data['entity_group']==ent_type]
            rows2drop = []

            # drop entities that do not have wikidata properties specific for the given entity type
            for idx in tqdm(sample.index.to_list()):
                qid = sample['wikidata_id'].loc[idx]
                if not any([wiki_prop in self.qid2attr[qid]['claims'].keys() for wiki_prop in self.wiki_prop4ent_type[ent_type]]):
                    rows2drop.append(idx)

                if ent_type == 'PER' and idx not in rows2drop:
                    # check if "instance of" has value "human" for entities labeled with type "PER"
                    instance_of_value = self.qid2attr[qid]['claims'][self.wiki_prop4ent_type[ent_type][0]][0]['mainsnak']['datavalue']['value']['id']
                    if instance_of_value != 'Q5':
                        rows2drop.append(idx)

            data = data.drop(rows2drop)
            log.info(f'Filtered {len(rows2drop)} entities which were incorrectly linked for type {ent_type}.')

        return data

    def _filter_on_nel_score(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        log.info('Filtering on named entity linking score.')
        init_data_len = len(data)
        
        data = data[data['nel_score'] >= threshold]
        log.info(f'Filtered {init_data_len - len(data)} entities.')

        return data

    def _filter_on_language(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info('Filtering on language.')
        qid2surface_form = self._map_qid2surface_form(data)

        foreign_labels = data[data['linked_entity_language'] != self.language]
        rows2drop = []

        for idx in tqdm(foreign_labels.index.to_list()):
            qid = foreign_labels['wikidata_id'].loc[idx]
            if len(qid2surface_form[qid]) == 1:
                rows2drop.append(idx)

        data = data.drop(rows2drop)
        log.info(f'Filtered {len(rows2drop)} entities.')

        return data

    def _filter_incorrect_author_type(self, data: pd.DataFrame, ent_type: str) -> pd.DataFrame:
        log.info(f'Filtering authors labeled as {ent_type}.')
        incorrect_sample = data[data['entity_group'] == ent_type].index
        
        data = data.drop(incorrect_sample)
        log.info(f'Filtered {len(incorrect_sample)} entities.')

        return data

    def _filter_authors(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info('Filtering authors without required keywords or properties in Wikidata.')
        correct_linked = []
        incorrect_linked = []

        for idx in tqdm(data.index.to_list()):
            row = data.loc[idx]
            qid = row['wikidata_id']
            ent_type = row['entity_group']

            if ent_type == 'PER':
                keyword_list = self.author_person_description
                author_values = self.author_person_occupation
                prop = 'P106'
            else:
                keyword_list = self.author_org_description
                author_values = self.author_org_instanceof
                prop = 'P31'

            if qid in self.qid2attr.keys():
                if self._check_property_values(qid, prop, author_values) or self._check_keyword_in_description(qid, keyword_list):
                    correct_linked.append(row)
                else:
                    incorrect_linked.append(row)
            else:
                incorrect_linked.append(row)

        correct_linked_authors = pd.DataFrame(correct_linked, columns=data.columns.to_list())
        incorrect_linked_authors = pd.DataFrame(incorrect_linked, columns=data.columns.to_list())

        # replace incorrect entity linking information with null values
        incorrect_linked_authors['linked_entity'] = incorrect_linked_authors['linked_entity'].apply(lambda _: None)
        incorrect_linked_authors['linked_entity_language'] = incorrect_linked_authors['linked_entity_language'].apply(lambda _: None)
        incorrect_linked_authors['nel_score'] = incorrect_linked_authors['nel_score'].apply(lambda _: None)
        incorrect_linked_authors['wikidata_id'] = incorrect_linked_authors['wikidata_id'].apply(lambda _: None)

        # concatenate the parsed authors
        data = pd.concat([correct_linked_authors, incorrect_linked_authors])
        log.info(f'Filtered {len(correct_linked_authors)} correctly and {len(incorrect_linked_authors)} incorrectly linked authors.')

        return data 

    def _check_property_values(self, qid: str, prop: str, values: Set[str]) -> bool:
        claims = self.qid2attr[qid]['claims']
        if prop in claims.keys():
            prop_claims = self.qid2attr[qid]['claims'][prop]
            prop_values = [prop['mainsnak']['datavalue']['value']['id'] for prop in prop_claims]
            return len(set(prop_values).intersection(values)) > 0
        return False

    def _check_keyword_in_description(self, qid: str, keyword_list: List[str])-> bool:
        description = WikidataItem(self.qid2attr[qid]).get_description()
        return any([keyword in description for keyword in keyword_list])
