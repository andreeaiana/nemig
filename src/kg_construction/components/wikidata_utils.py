from typing import Any, Set, List, Dict, Union, Tuple

from tqdm import tqdm
from qwikidata.entity import WikidataItem
from qwikidata.typedefs import EntityDict
from qwikidata.linked_data_interface import get_entity_dict_from_api

import src.kg_construction.components.graph_utils as kg_utils
import src.utils.data_utils as data_utils
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def get_wiki_label(attr_dict: EntityDict) -> Union[str, None]:
    try:
        item = WikidataItem(attr_dict)
        label = item.get_label()
        if label == '':
            label = None
    except Exception:
        label = None
    
    return label


def update_qid2attr_map(qid2attr: Dict[str, Any], qid: str) -> Dict[str, Any]:
    if qid not in qid2attr.keys():
        try:
            attr_dict = get_entity_dict_from_api(qid)
            qid2attr[qid] = attr_dict
        except Exception:
            pass

    return qid2attr


def retrieve_wikidata_neighbors(
        entities: List[str], 
        qid2attr: Dict[str, Any], 
        qid2attr_filepath: str, 
        wiki_neighbors_map: Dict[str, Set[Tuple[str, str, str]]],
        wiki_neighbors_map_filepath: str,
        wikidata_property_namespace: str,
        wikidata_resource_namespace: str
        ) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]]]:
    log.info(f'Retrieving neighbors from Wikidata for {len(entities)} entities.')   
    
    # find entity neighbors
    triples = set()
    steps = 0
    log.info(f'Extracting neighbors for {len(entities)} entities.')
    for entity in tqdm(entities):
        if entity in wiki_neighbors_map.keys():
            # entity is already in the map, directly retrieve it    
            triples.update(wiki_neighbors_map[entity])
        else:
            if entity in qid2attr.keys():
                data = qid2attr[entity]
                claims = data['claims']
                properties = [prop for prop in claims]
                for prop in properties:
                    for item in claims[prop]:
                        if (('datavalue' in item['mainsnak']) and (type(item['mainsnak']['datavalue']['value'])==dict) and ('id' in item['mainsnak']['datavalue']['value'])):
                            triple = (entity, kg_utils.pid2wikidata_property(prop, wikidata_property_namespace), kg_utils.qid2wikidata_resource(item['mainsnak']['datavalue']['value']['id'], wikidata_resource_namespace))
                            triples.add(triple)
                            wiki_neighbors_map[entity].add(triple)

        # cache wiki_neighbors_map 
        if steps % 50000 == 0:
            data_utils.update_cache(wiki_neighbors_map, wiki_neighbors_map_filepath)
        steps += 1
    data_utils.update_cache(wiki_neighbors_map, wiki_neighbors_map_filepath)

    log.info(f'Retrieved {len(triples)} neighbors.')
    log.info(f'Size of qid2attr map: {len(qid2attr)}.')

    # retrieve neighbors' information
    querying_steps = 0
    target_entities = set([triple[2] for triple in triples])
    target_entities = [kg_utils.wikidata_resource2qid(entity, wikidata_resource_namespace) for entity in target_entities]

    log.info(f'Querying Wikidata for neighbor information for {len(set(target_entities))} entities.')
    for entity in tqdm(list(set(target_entities))):
        qid2attr = update_qid2attr_map(qid2attr, entity)
        if querying_steps % 50000 == 0:
            data_utils.update_cache(qid2attr, qid2attr_filepath)
        querying_steps += 1
    data_utils.update_cache(qid2attr, qid2attr_filepath)

    return qid2attr, list(triples)
