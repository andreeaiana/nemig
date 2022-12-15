from typing import Set, List, Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from difflib import SequenceMatcher
from collections import defaultdict

from src.kg_construction.components.base_graph import BaseGraph
import src.kg_construction.components.normalize_util as norm_utils
import src.kg_construction.components.graph_utils as kg_utils
import src.kg_construction.components.wikidata_utils as wiki_utils
import src.utils.data_utils as data_utils
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class NewsKG(BaseGraph):
    def __init__(
            self,
            namespaces: Dict[str, str],
            predicates: Dict[str, str],
            classes: Dict[str, str],
            overwrite_cache: bool,
            graph_type: str,
            min_unlinked_resource_freq: int,
            min_sink_entities_freq: int,
            related_news_threshold: float,
            k_hop: int,
            attributes: Dict[str, str],
            node_prefix: Dict[str, str],
            data2include: Dict[str, str],
            input_files: Dict[str, str],
            cache_files: Dict[str, str]
            ) -> None:
        
        super().__init__()

        self.namespaces = namespaces
        self.predicates = predicates
        self.classes = classes

        self.overwrite_cache = overwrite_cache
        self.graph_type = graph_type
        self.min_unlinked_resource_freq = min_unlinked_resource_freq
        self.min_sink_entities_freq = min_sink_entities_freq
        self.related_news_threshold = related_news_threshold
        self.k_hop = k_hop
        self.attributes = attributes

        self.node_prefix = node_prefix
        self.data2include = data2include

        self.input_files = input_files
        self.cache_files = cache_files

        assert all([data_utils.check_integrity(file) for file in self.input_files.values()])
        
        # Initialize graph
        self.graph = nx.MultiDiGraph()

    def _check_node_exists(self, node: str) -> None:
        if not self.has_node(node):
            raise Exception(f'Node {node} not in graph')

    def get_node_label(self, node: str) -> str:
        """ Returns the label of a node. """
        if not (kg_utils.is_custom_resource(node, self.namespaces['custom_resource']) or kg_utils.is_wikidata_resource(node, self.namespaces['wikidata_resource'])):
            raise Exception(f'Node {node} is not a resource and does not have a label')
        
        self._check_node_exists(node)
        return self._get_attr(node, self.attributes['label'])

    def set_node_label(self, node: str, label: str) -> None:
        """ Sets the label of a node. """
        if not (kg_utils.is_custom_resource(node, self.namespaces['custom_resource']) or kg_utils.is_wikidata_resource(node, self.namespaces['wikidata_resource'])):
            raise Exception(f'Node {node} is not a resource and cannot have a label.')
        
        if label == None:
            self._set_attr(node, self.attributes['label'], label)
        else:
            if not len(label) == 1:
                self._set_attr(node, self.attributes['label'], label)

    def set_nodes_labels(self, nodes_w_labels: Tuple[str, str]) -> None:
        """ Sets the label of a node. """
        for i in range(len(nodes_w_labels)):
            self._check_node_exists(nodes_w_labels[i][0])
            self.set_node_label(node=nodes_w_labels[i][0], label=nodes_w_labels[i][1])

    def get_edges_by_key(self, key: str) -> Set[Tuple[str, str]]:
        """ Returns all edges with the given key contained in the graph. """
        edges = [(u, v) for u, v, k in self.get_edges(keys=True) if k==key]
        return set(edges)
    
    def get_custom_resources(self) -> Set[str]:
        """ Returns all the GeNeG resources contained in the graph. """
        custom_resources = [node for node in self.nodes if kg_utils.is_custom_resource(node, self.namespaces['custom_resource'])]
        return set(custom_resources)

    def get_wikidata_resources(self) -> Set[str]:
        """ Returns all entities linked to Wikidata. """
        wikidata_resources = [node for node in self.nodes if kg_utils.is_wikidata_resource(node, self.namespaces['wikidata_resource'])]
        return set(wikidata_resources)
    
    def get_unlinked_resources(self) -> Set[str]:
        """ Returns any resource that is not linked to Wikidata and that does not represent an article, event, or content node. """
        resource_nodes = self.get_custom_resources()
        news_nodes = set(self.get_news_nodes())
        content_nodes = set(self.get_content_nodes())
        event_nodes = set(self.get_event_nodes())
        topic_nodes = set(self.get_topic_nodes())

        other_nodes = news_nodes.union(content_nodes).union(event_nodes).union(topic_nodes)
        return resource_nodes.difference(other_nodes)

    def get_news_nodes(self) -> List[str]:
        """ Returns the ids of all nodes representing a resource corresponding to a news article. """
        blank_nodes = [node for node in self.get_custom_resources() if 'news_' in node]
        content_nodes = set(self.get_content_nodes())
        event_nodes = set(self.get_event_nodes())
        topic_nodes = set(self.get_topic_nodes())
        
        news_nodes = set(blank_nodes).difference(content_nodes).difference(event_nodes).difference(content_nodes).difference(topic_nodes)
        return list(news_nodes)

    def get_event_nodes(self) -> List[str]:
        """ Returns the ids of all nodes representing a resource corresponding to an event. """
        event_nodes = [node for node in self.get_custom_resources() if '_evt' in node]
        return event_nodes

    def get_topic_nodes(self) -> List[str]:
        """ Returns the ids of all nodes representing a resource corresponding to a topic. """
        topic_nodes = [node for node in self.get_custom_resources() if '_topic' in node]
        return topic_nodes

    def get_content_nodes(self, content_part: str = None) -> List[str]:
        """ Returns the ids of all nodes representing a resource corresponding to a content part. """
        if content_part == None:
            # retrieve all content nodes
            content_nodes = []
            for content_part in [self.node_prefix['news_title'], self.node_prefix['news_abstract'], self.node_prefix['news_body']]:
                content_nodes.extend([node for node in self.get_custom_resources() if content_part in node])
        else:
            # retrieve content nodes only for content_part
            content_nodes = [node for node in self.get_custom_resources() if content_part in node]

        return content_nodes

    def get_all_properties(self) -> Set[str]:
        """ Returns all properties used in the graph. """
        properties = [k for _, _,k in self.get_edges(keys=True)]
        return set(properties)

    def get_properties_for_node(self, node: str) -> Set[str]:
        """ Returns all properties that the given node has (i.e. given node is the source node of the edge). """
        properties = [k for u, _, k in self.get_edges(keys=True) if u==node]
        return set(properties)

    def get_subjects_for_property(self, prop: str) -> Set[str]:
        """ Returns all subject nodes that have the given property (i.e. all source nodes for edges of type 'prop'). """
        source_nodes = [u for u, _, k in self.get_edges(keys=True) if k==prop]
        return set(source_nodes)

    def get_objects_for_property(self, prop: str) -> Set[str]:
        """ Returns all objects of a given property (i.e. all target nodes for edges of type 'prop'). """
        objects = [v for _, v, k in self.get_edges(keys=True) if k==prop]
        return set(objects)

    def get_nodes_for_property(self, prop: str) -> Set[Tuple[str, str]]:
        """ Returns all pairs of nodes sharing an edge of type 'prop'. """
        nodes = [(u, v) for u, v, k in self.get_edges(keys=True) if k==prop]
        return set(nodes)

    def get_subjects(self, prop: str, target_node: str) -> Set[str]:
        """ Returns all source nodes for the given property and target node. """
        source_nodes = [u for u, v, k in self.get_edges(keys=True) if k==prop and v==target_node]
        return set(source_nodes)

    def get_objects(self, prop: str, source_node: str) -> Set[str]:
        """ Returns all target nodes for the given property and source node. """
        target_nodes = [v for u, v, k in self.get_edges(keys=True) if k==prop and u==source_node]
        return set(target_nodes)

    def get_source_for_target(self, target_node: str) -> Set[str]:
        """ Returns, for the given target node, all source nodes for all properties. """
        source_nodes = [u for u, v, _ in self.get_edges(keys=True) if v==target_node]
        return set(source_nodes)

    def get_target_for_source(self, source_node: str) -> Set[str]:
        """ Returns, for the given source node, all target nodes for all properties. """
        target_nodes = [v for u, v, _ in self.get_edges(keys=True) if u==source_node]
        return set(target_nodes)

    def remove_subjecs_for_property(self, prop: str) -> None:
        """ Removes all source nodes with an edge of type 'prop' from the graph. """
        remove_nodes = self.get_subjects_for_property(prop)
        self._remove_nodes(remove_nodes)

    def remove_edges_for_property(self, prop: str) -> None:
        """ Removes all edges of type 'prop' from the graph. """
        remove_edges = self.get_edges_for_property(prop)
        self._remove_edges(remove_edges)

    @property
    def statistics(self) -> str:
        # Frequency of different types of nodes
        custom_resources = self.get_custom_resources()
        unlinked_custom_resources = self.get_unlinked_resources()
        wikidata_resources = self.get_wikidata_resources()

        count_resources = len(custom_resources) + len(wikidata_resources)
        count_literals = len(self.nodes) - count_resources
        count_blank_resources = len(custom_resources) - len(unlinked_custom_resources) 

        return '\n'.join([
            '{:^43}'.format('STATISTICS'),
            '=' * 43,
            '\n',
            '{:^43}'.format('General'),
            '-' * 43,
            '{:<30} | {:>10}'.format('nodes', len(self.nodes)),
            '{:<30} | {:>10}'.format('edges', len(self.edges)),
            '\n',

            '-' * 43,
            '{:^43}'.format('Nodes'),
            '-' * 43,

            '{:<30} | {:>10}'.format('All nodes', len(self.nodes)),
            '\n',
            '{:<30}   {:>10}'.format('', '-frequency-'),
            '  {:<28} | {:>10}'.format('resources', round(count_resources/len(self.nodes), 4)),
            '  {:<28} | {:>10}'.format('literals', round(count_literals/len(self.nodes), 4)),

            '\n',
            '{:<30} | {:>10}'.format('Resources', count_resources),
            '\n',
            '{:<30}   {:>10}'.format('', '-frequency-'),
            '  {:<28} | {:>10}'.format('Blank resource nodes', round(count_blank_resources / count_resources, 4)),
            '  {:<28} | {:>10}'.format('NeMig resources', round(len(unlinked_custom_resources) / count_resources, 4)),
            '  {:<28} | {:>10}'.format('Wikidata resources', round(len(wikidata_resources) / count_resources, 4)),
            '\n',
            ])

    def get_graph(self):
        graph_filepath = self.cache_files['graph'][self.graph_type]
        if data_utils.check_integrity(graph_filepath) and not self.overwrite_cache:
            log.info(f'News graph of type {self.graph_type} already cached at {graph_filepath}. Loading the graph into memory.')
            self.graph = data_utils.load_cache(graph_filepath)
        else:
            log.info(f'News graph of type {self.graph_type} not built yet. Constructing the graph.')        
            self.graph = self.construct_graph()

        return self.graph

    def construct_graph(self):
        log.info(f'Constructing {self.graph_type} news knowledge graph.')
        
        log.info(f'Loading qid2attr mapping.')
        self.qid2attr = data_utils.load_cache(self.cache_files['wikidata']['qid2attr_map'])

        if self.graph_type == 'base':
            self._construct_base_graph()

            # Remove infrequent unlinked and sink entities
            self._remove_infrequent_unlinked_resources()
            self._remove_sink_entities()

        elif 'entities' in self.graph_type:
            """ Base graph without literals, optionally with k-hop Wikidata neighbors (enriched). """
            enrich_flag = True if self.graph_type=='enriched_entities' else False
            self._contruct_entities_graph(remove_literals=True, enrich_graph=enrich_flag)
            
            # Remove infrequent unlinked and sink entities
            self._remove_infrequent_unlinked_resources()
            self._remove_sink_entities()

        elif self.graph_type == 'complete':
            """ Base graph (entities & literals) + k-hop Wikidata neighbors. """
            self._contruct_entities_graph(remove_literals=False, enrich_graph=True)
            
            # Remove infrequent unlinked and sink entities
            self._remove_infrequent_unlinked_resources()
            self._remove_sink_entities()

        else:
            raise ValueError(f'Graph type {self.graph_type} is unknown.')

        log.info(f'Finalized construction of graph type {self.graph_type}: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges.')

        # Cache updated qid2attr mapping to disk
        log.info('Caching qid2attr map to disk.')
        data_utils.update_cache(self.qid2attr, self.cache_files['wikidata']['qid2attr_map'])

        # Cache graph to disk
        log.info('Caching graph to disk.')
        data_utils.update_cache(self.graph, self.cache_files['graph'][self.graph_type]) 

    def _construct_base_graph(self):
        # Load general dataset
        data = pd.read_csv(self.input_files['general'])
        log.info(f'Loaded dataset with {len(data)} articles.')

        # Populate with empty nodes
        assert 'id' in self.data2include.keys()
        
        # news nodes
        news_node_ids = [self._news_id2node_id(id, prefix=self.node_prefix['news']) for id in data[self.data2include['id']]]
        self._add_nodes(news_node_ids)
        self._add_class_type_relation(source_nodes=news_node_ids, class_type=self.classes['news_article'])        
        news_indices = [self._node_id2news_idx(data, node, prefix=self.node_prefix['news']) for node in news_node_ids]
        
        # provenance information (e.g. URLs)
        if 'url' in self.data2include.keys():
            urls = [data.loc[idx][self.data2include['url']].values[0] for idx in news_indices]
            self._add_simple_node_relation(source_nodes=news_node_ids, target_nodes=urls, property_type=self.predicates['url'])
        
        # topic information
        if 'topic' in self.data2include.keys():
            topics = data[self.data2include['topic']].unique().tolist()
            topics2node_ids = dict()
            for topic_id in topics:
                topics2node_ids[topic_id] = self._news_id2node_id(topic_id, prefix=self.node_prefix['news_topic']) 
            topic_ids = list(topics2node_ids.values())
            self._add_nodes(topic_ids)

            topics_nodes = [topics2node_ids[topic] for topic in data[self.data2include['topic']].tolist()]
            self._add_simple_node_relation(source_nodes=news_node_ids, target_nodes=topics_nodes, property_type=self.predicates['subject'])

            if 'topic_label' in self.data2include.keys():
                # preprocess entity data
                topic_label_data = pd.read_csv(self.input_files['topic_label'])
                topic_label_data['wikidata_id'] = topic_label_data['wikidata_id'].fillna('')
                topic_label_data['entity'] = topic_label_data.apply(
                        lambda row: kg_utils.qid2wikidata_resource(
                            row['wikidata_id'], self.namespaces['wikidata_resource']
                            ) if row['wikidata_id']!='' else kg_utils.label2custom_resource(
                            row[self.data2include['topic_label']].strip('[START] ').strip(' [END]'), self.namespaces['custom_resource']
                            ),
                        axis=1
                        )
                topic_label_data['entity_label'] = topic_label_data.apply(
                        lambda row: norm_utils.get_canonical_label(row['linked_entity']) if row['wikidata_id']!='' else
                        norm_utils.get_canonical_label(norm_utils.clean_label(row[self.data2include['topic_label']].strip('[START] ').strip(' [END]'), entity_type='')),
                        axis=1
                        )

                topics2news_id = data.groupby(self.data2include['topic'])[self.data2include['id']].apply(list).reset_index()
                topics2news_id[self.data2include['id']] = topics2news_id[self.data2include['id']].apply(lambda x: x[0])
                news_id2topic = pd.Series(topics2news_id[self.data2include['topic']].values, index=topics2news_id[self.data2include['id']]).to_dict()
                topic_label_sample = topic_label_data[topic_label_data[self.data2include['id']].isin(news_id2topic.keys())]

                edges = list()
                for news_id in topic_label_sample[self.data2include['id']].unique():
                    topic_keywords = topic_label_sample[topic_label_sample[self.data2include['id']]==news_id]['entity'].tolist()
                    topic = topics2node_ids[news_id2topic[news_id]]
                    edges.extend([(topic, keyword) for keyword in topic_keywords])

                self._add_entity_node_relation(edges, property_type=self.predicates['keywords'])
                    
                topics_w_labels = list(set(zip(
                    topic_label_data['entity'].tolist(), 
                    topic_label_data['entity_label'].tolist()
                    )))
                self.set_nodes_labels(topics_w_labels)
        
        # publisher information
        if 'publisher' in self.data2include.keys():
            # load publisher data
            publisher_data = pd.read_csv(self.input_files['disambiguated_publisher'])
            disambiguated_publishers = self._add_publisher_relation(data, news_node_ids, publisher_data)           
        
            if 'political_orientation' in self.data2include.keys():
                political_orientation_data = pd.read_csv(self.input_files['political_orientation'])
                political_orientation_data = political_orientation_data.set_index(self.data2include['publisher'], drop=True)
                outlet2pol_orient = political_orientation_data.to_dict('index')
                
                source_nodes = []
                target_nodes = []
                for outlet in disambiguated_publishers.keys():
                    source_nodes.append(disambiguated_publishers[outlet]['resource'])
                    target_nodes.append(outlet2pol_orient[outlet]['political_orientation'])
                
                self._add_simple_node_relation(source_nodes=source_nodes, target_nodes=target_nodes, property_type=self.predicates['political_orientation'])

        # date information (e.g. data published and modified)
        if 'date_published' in self.data2include.keys():
            dates_published = [data.loc[idx][self.data2include['date_published']].values[0] for idx in news_indices]
            self._add_simple_node_relation(source_nodes=news_node_ids, target_nodes=dates_published, property_type=self.predicates['date_published'])

        if 'date_modified' in self.data2include.keys():
            dates_modified = [data.loc[idx][self.data2include['date_modified']].values[0] for idx in news_indices]
            self._add_simple_node_relation(source_nodes=news_node_ids, target_nodes=dates_modified, property_type=self.predicates['date_modified'])
        
        # sentiment labels
        if 'sentiment' in self.data2include.keys():
            # load sentiment annotations
            sentiment_data = pd.read_csv(self.input_files['sentiment'])
           
            # add sentiment annotations
            sentiment_news_indices = [self._node_id2news_idx(sentiment_data, node, prefix=self.node_prefix['news']) for node in news_node_ids]
            sentiment_labels = [sentiment_data.loc[idx][self.data2include['sentiment']].values[0] for idx in sentiment_news_indices]
            self._add_simple_node_relation(source_nodes=news_node_ids, target_nodes=sentiment_labels, property_type=self.predicates['sentiment'])

        # author information
        if 'author' in self.data2include.keys():
            author_data = pd.read_csv(self.input_files['author'])
            self._add_metadata_entity_relation(news_node_ids, self.predicates['author'], author_data, self.data2include['author'])

        if 'author_org' in self.data2include.keys():
            author_org_data = pd.read_csv(self.input_files['author_org'])
            self._add_metadata_entity_relation(news_node_ids, self.predicates['author'], author_org_data, self.data2include['author_org'])

        # keyword information
        if 'keywords' in self.data2include.keys():
            keywords_data = pd.read_csv(self.input_files['keywords'])
            self._add_metadata_entity_relation(news_node_ids, self.predicates['keywords'], keywords_data, self.data2include['keywords'])

        # check if news are based on one another
        related_news_ids = self._get_related_articles(data) 
        source_related_news = [self._retrieve_node_id_from_news_id(ids[0], prefix=self.node_prefix['news']) for ids in related_news_ids]
        target_related_news = [self._retrieve_node_id_from_news_id(ids[1], prefix=self.node_prefix['news']) for ids in related_news_ids]
        self._add_simple_node_relation(source_nodes=source_related_news, target_nodes=target_related_news, property_type=self.predicates['is_based_on'])

        # event nodes 
        event_node_ids = [self._news_id2node_id(id, prefix=self.node_prefix['news_event']) for id in data[self.data2include['id']]]
        self._add_simple_node_relation(source_nodes=news_node_ids, target_nodes=event_node_ids, property_type=self.predicates['about'])
        self._add_class_type_relation(source_nodes=event_node_ids, class_type=self.classes['event'])

        if 'title' in self.data2include.keys():
            # content nodes
            title_nodes = [self._news_id2node_id(id, prefix=self.node_prefix['news_title']) for id in data[self.data2include['id']]]
            self._add_simple_node_relation(source_nodes=news_node_ids, target_nodes=title_nodes, property_type=self.predicates['has_part'])
            
            # literals
            titles = [data.loc[idx][self.data2include['title']].values[0] for idx in news_indices]
            self._add_simple_node_relation(source_nodes=title_nodes, target_nodes=titles, property_type=self.predicates['headline'])
            
            # entities
            title_entities = pd.read_csv(self.input_files['title_entities'])
            self._add_content_entity_relation(event_node_ids, title_nodes, 'news_title', title_entities)

        if 'abstract' in self.data2include.keys():
            # content nodes
            data[self.data2include['abstract']] = data[self.data2include['abstract']].fillna("")
            news_w_abstract = [news_node_ids[i] for i in range(len(news_indices)) if data.loc[news_indices[i]][self.data2include['abstract']].values[0]!=""]
            abstract_nodes = [self._news_id2node_id(self._node_id2news_id(id, prefix=self.node_prefix['news']), prefix=self.node_prefix['news_abstract']) for id in news_w_abstract]
            self._add_simple_node_relation(source_nodes=news_w_abstract, target_nodes=abstract_nodes, property_type=self.predicates['has_part'])
            
            # literals
            abstract_indices = [self._node_id2news_idx(data, node, prefix=self.node_prefix['news_abstract']) for node in abstract_nodes]
            abstracts = [data.loc[idx][self.data2include['abstract']].values[0] for idx in abstract_indices]
            self._add_simple_node_relation(source_nodes=abstract_nodes, target_nodes=abstracts, property_type=self.predicates['abstract'])
            
            # entities
            abstract_entities = pd.read_csv(self.input_files['abstract_entities'])
            self._add_content_entity_relation(event_node_ids, abstract_nodes, 'news_abstract', abstract_entities)

        if 'body' in self.data2include.keys():
            # content nodes
            body_nodes = [self._news_id2node_id(id, prefix=self.node_prefix['news_body']) for id in data[self.data2include['id']]]
            self._add_simple_node_relation(source_nodes=news_node_ids, target_nodes=body_nodes, property_type=self.predicates['has_part'])
            
            # literals
            bodies = [data.loc[idx][self.data2include['body']].values[0] for idx in news_indices]
            self._add_simple_node_relation(source_nodes=body_nodes, target_nodes=bodies, property_type=self.predicates['article_body'])
            
            # entities
            body_entities = pd.read_csv(self.input_files['body_entities'])
            self._add_content_entity_relation(event_node_ids, body_nodes, 'news_body', body_entities)
        
    def _contruct_entities_graph(self, remove_literals: bool, enrich_graph: bool):
        # Construct base graph
        self._construct_base_graph() 

        if remove_literals:
            # Remove literals from base graph
            log.info('Removing literals from graph.')
            resources = self.get_custom_resources().union(self.get_wikidata_resources())
            literals = self.nodes.difference(resources)
            log.info(f'Found {len(literals)} literal nodes.')
            self._remove_nodes(literals)
            log.info(f'New graph size: {len(self.nodes)} nodes, {len(self.edges)} edges.')

        # If graph type is enriched, add Wikidata neighbors
        if enrich_graph:
            log.info('Enriching graph with k-hop neighbor entities from Wikidata.')
            self._add_k_hop_neighbors()

    def _add_k_hop_neighbors(self) -> None:
        for k in range(1, self.k_hop + 1):
            triples = self._get_k_hop_neighbors(k)
            source_nodes = list(set([triple[0] for triple in triples]))
            target_nodes = list(set([triple[2] for triple in triples]))
            log.info(f'Extending graph with {len(triples)} triples representing {len(target_nodes)} {k}-hop neighbors of {len(source_nodes)} Wikidata-linked entities.')

            # add edges
            edge_types = set([triple[1] for triple in triples])
            for edge_type in edge_types:
                edges = list(set([(kg_utils.qid2wikidata_resource(triple[0], self.namespaces['wikidata_resource']), triple[2], triple[1]) for triple in triples if triple[1]==edge_type]))
                self._add_edges(edges)
            log.info(f'New graph size: {len(self.nodes)} nodes, {len(self.edges)} edges.')

            # add labels
            log.info('Adding node labels.')
            target_labels = [norm_utils.get_canonical_label(
                wiki_utils.get_wiki_label(
                    self.qid2attr[kg_utils.wikidata_resource2qid(qid, self.namespaces['wikidata_resource'])]
                    )
                ) if kg_utils.wikidata_resource2qid(qid, self.namespaces['wikidata_resource']) in self.qid2attr.keys() else None for qid in target_nodes]
            self.set_nodes_labels(list(zip(target_nodes, target_labels)))
            log.info('Finished adding node labels.')

    def _get_k_hop_neighbors(self, k: int) -> List[Tuple[str, str, str]]:
        if k == 1:
            entities = list(self.get_wikidata_resources())
        else:
            if not data_utils.check_integrity(self.cache_files['wikidata']['hop_neighbors_list'][k-1]):
                raise Exception(f'{k-1} neighbors need to be computed first.')

            triples = data_utils.load_cache(self.cache_files['wikidata']['hop_neighbors_list'][k-1])
            entities = set([triple[2] for triple in triples])

        entities = [kg_utils.wikidata_resource2qid(entity, self.namespaces['wikidata_resource']) for entity in entities]

        if data_utils.check_integrity(self.cache_files['wikidata']['neighbors_map']):
            wiki_neighbors_map = data_utils.load_cache(self.cache_files['wikidata']['neighbors_map'])
        else:
            wiki_neighbors_map = defaultdict(set)

        self.qid2attr, wiki_triples = wiki_utils.retrieve_wikidata_neighbors(
                entities = entities, 
                qid2attr = self.qid2attr, 
                qid2attr_filepath = self.cache_files['wikidata']['qid2attr_map'],
                wiki_neighbors_map = wiki_neighbors_map,
                wiki_neighbors_map_filepath=self.cache_files['wikidata']['neighbors_map'],
                wikidata_property_namespace=self.namespaces['wikidata_property'],
                wikidata_resource_namespace=self.namespaces['wikidata_resource']
                )
        data_utils.update_cache(wiki_triples, self.cache_files['wikidata']['hop_neighbors_list'][k])

        return wiki_triples

    def _news_id2node_id(self, data_id: int, prefix: str) -> str:
        """ Generate a unique node id from its corresponding data id. """
        node_id = kg_utils.id2custom_resource(prefix + str(data_id), self.namespaces['custom_resource'])
        if self.has_node(node_id):
            raise Exception

        return node_id

    def _retrieve_node_id_from_news_id(self, data_id: int, prefix: str) -> str:
        node_id = kg_utils.id2custom_resource(prefix + str(data_id), self.namespaces['custom_resource'])
        if not self.has_node(node_id):
            raise Exception

        return node_id

    def _node_id2news_id(self, node_id: str, prefix: str) -> int:
        """ Returns the data id of the news article with the given node id """
        self._check_node_exists(node_id)
        news_id = kg_utils.custom_resource2id(node_id, self.namespaces['custom_resource'])
        news_id = news_id.split(prefix)[-1]
        return int(news_id)

    def _node_id2news_idx(self, data: pd.DataFrame, node_id: str, prefix: str) -> List[int]:
        self._check_node_exists(node_id)
        news_indices = data[data[self.data2include['id']] == self._node_id2news_id(node_id ,prefix)].index.tolist()
        return news_indices

    def _add_class_type_relation(self, source_nodes: List[str], class_type: str) -> None:
        target_nodes = [class_type] * len(source_nodes)
        self._add_simple_node_relation(source_nodes=source_nodes, target_nodes=target_nodes, property_type=self.predicates['type'])

    def _add_simple_node_relation(self, source_nodes: List[str], target_nodes: List[str], property_type: str) -> None:
        """ Adds edges of type 'property' between the source and target nodes, where the target nodes are all of the same kind."""
        log.info(f'Populating NewsKG with edges of type {property_type}.')
        assert len(source_nodes) == len(target_nodes)

        edge_type = [property_type] * len(source_nodes)
        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        log.info(f'New graph size: {len(self.nodes)} nodes, {len(self.edges)} edges.')

    def _add_entity_node_relation(self, edges: Tuple[str, str], property_type: str) -> None:
        """ Adds edges of type 'property' between the edge nodes, where the target nodes can be of 2 kinds, i.e. linked and non-linked resources."""
        log.info(f'Populating NewsKG with edges of type {property_type}.')
        
        edges = [(edge[0], edge[1], property_type) for edge in list(set(edges))]
        self._add_edges(edges)
        log.info(f'New graph size: {len(self.nodes)} nodes, {len(self.edges)} edges.')

    def _add_content_entity_relation(self, all_source_nodes: List[str], all_content_nodes: List[str], content_node_prefix: str, entity_data: pd.DataFrame) -> None:
        source_nodes = [node_id for node_id in all_source_nodes if self._node_id2news_id(node_id, self.node_prefix['news_event']) in entity_data[self.data2include['id']]]
        target_indices = [self._node_id2news_idx(entity_data, node_id, self.node_prefix['news_event']) for node_id in source_nodes]
        target_indices = [idx if type(idx)==list else [idx] for idx in target_indices]
        entity_nodes = [entity_data[['wikidata_id', 'linked_entity', 'entity_group']].loc[idx].values.tolist() for idx in target_indices]
        edges = [(source_nodes[i], entity) for i in range(len(source_nodes)) for entity in entity_nodes[i]]
        edges = [(edge[0], [kg_utils.qid2wikidata_resource(edge[1][0], self.namespaces['wikidata_resource']), edge[1][1], edge[1][2]]) for edge in edges]

        # PER & ORG entities
        actor_edges = [(edge[0], edge[1][0]) for edge in edges if edge[1][2]=='PER' or edge[1][2]=='ORG']
        actor_entity_labels = [(edge[1][0], norm_utils.get_canonical_label(edge[1][1])) for edge in edges if edge[1][2]=='PER' or edge[1][2]=='ORG']
        self._add_entity_node_relation(actor_edges, self.predicates['has_actor'])
        self.set_nodes_labels(actor_entity_labels) 

        # LOC entities
        place_edges = [(edge[0], edge[1][0]) for edge in edges if edge[1][2]=='LOC']
        place_entity_labels = [(edge[1][0], norm_utils.get_canonical_label(edge[1][1])) for edge in edges if edge[1][2]=='LOC']
        self._add_entity_node_relation(place_edges, self.predicates['has_place'])
        self.set_nodes_labels(place_entity_labels) 

        # MISC entities
        mention_edges = [(edge[0], edge[1][0]) for edge in edges if edge[1][2]=='MISC']
        mention_entity_labels = [(edge[1][0], norm_utils.get_canonical_label(edge[1][1])) for edge in edges if edge[1][2]=='MISC']
        self._add_entity_node_relation(mention_edges, self.predicates['has_place'])
        self.set_nodes_labels(mention_entity_labels) 

        # Relation: entity -> isReferencedBy -> title_node
        target_content_nodes = [node_id for node_id in all_content_nodes if self._node_id2news_id(node_id, self.node_prefix[content_node_prefix]) in entity_data[self.data2include['id']]]
        referenced_edges = [(kg_utils.qid2wikidata_resource(entity[0], self.namespaces['wikidata_resource']), target_content_nodes[i]) for i in range(len(target_content_nodes)) for entity in entity_nodes[i]]
        self._add_entity_node_relation(referenced_edges, self.predicates['is_referenced_by'])

    def _add_metadata_entity_relation(self, all_source_nodes: List[str], relationship: str, entity_data: pd.DataFrame, data_col: str) -> None:
        entity_data['wikidata_id'] = entity_data['wikidata_id'].fillna('')
        entity_data['entity_group'] = entity_data['entity_group'].fillna('')

        source_nodes = [node_id for node_id in all_source_nodes if self._node_id2news_id(node_id, self.node_prefix['news']) in entity_data[self.data2include['id']]]
        target_indices = [self._node_id2news_idx(entity_data, node_id, self.node_prefix['news']) for node_id in source_nodes]
        target_indices = [idx if type(idx)==list else [idx] for idx in target_indices]
        entity_nodes = [entity_data[['wikidata_id', 'linked_entity', 'entity_group', data_col]].loc[idx].values.tolist() for idx in target_indices]
        edges = [(source_nodes[i], entity) for i in range(len(source_nodes)) for entity in entity_nodes[i]]
        
        # Relations between news and disambiguated entities linked to Wikidata
        linked_edges = [(edge[0], kg_utils.qid2wikidata_resource(edge[1][0], self.namespaces['wikidata_resource'])) for edge in edges if edge[1][0]!='']
        if relationship == self.predicates['keywords']: 
            linked_entity_labels = [(kg_utils.qid2wikidata_resource(edge[1][0], self.namespaces['wikidata_resource']), norm_utils.get_canonical_label(edge[1][1])) for edge in edges if edge[1][0]!='']
        else:
            linked_entity_labels = [(kg_utils.qid2wikidata_resource(edge[1][0], self.namespaces['wikidata_resource']), wiki_utils.get_wiki_label(self.qid2attr[edge[1][0]])) for edge in edges if edge[1][0]!='']
            linked_entity_labels = [(edge[0], norm_utils.get_canonical_label(edge[1])) if edge[1]!=None else (edge[0], None) for edge in linked_entity_labels]
        self._add_entity_node_relation(linked_edges, relationship)
        self.set_nodes_labels(linked_entity_labels)

        # Relations between news and non-disambiguated entities not linked to Wikidata
        if relationship == self.predicates['keywords']:
            non_linked_edge_labels = [(edge[0], norm_utils.get_canonical_label(norm_utils.clean_label(edge[1][1], entity_type=edge[1][2]))) for edge in edges]
        else:
            non_linked_edge_labels = [(edge[0], norm_utils.get_canonical_label(norm_utils.clean_label(edge[1][3].strip('[START] ').strip(' [END]'), entity_type=''))) for edge in edges if edge[1][0]=='']
        non_linked_edges = [(edge[0], kg_utils.label2custom_resource(edge[1], self.namespaces['custom_resource'])) for edge in non_linked_edge_labels]
        non_linked_entities_labels = [(kg_utils.label2custom_resource(edge[1], self.namespaces['custom_resource']), edge[1]) for edge in non_linked_edge_labels]
        self._add_entity_node_relation(non_linked_edges, relationship)
        self.set_nodes_labels(non_linked_entities_labels)

    def _add_publisher_relation(self, data: pd.DataFrame, source_nodes: List[str], entity_data: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        entity_data['entity'] = entity_data['entity'].fillna('')
        entity_data['wikidata_id'] = entity_data['wikidata_id'].fillna('')
        entity_data = entity_data.set_index(self.data2include['publisher'], drop=True)
        disambiguated_publishers = entity_data.to_dict('index')

        # Update qid2attr with wikidata information
        publishers_qids = [qid for qid in entity_data['wikidata_id'].unique().tolist() if qid!='']
        for qid in tqdm(publishers_qids):
            self.qid2attr = wiki_utils.update_qid2attr_map(self.qid2attr, qid)

        for outlet in disambiguated_publishers.keys():
            qid = disambiguated_publishers[outlet]['wikidata_id']
            if qid!='':
                label = norm_utils.get_canonical_label(wiki_utils.get_wiki_label(self.qid2attr[qid]))
                resource = kg_utils.qid2wikidata_resource(qid, self.namespaces['wikidata_resource'])
            else:
                label = norm_utils.get_canonical_label(norm_utils.clean_label(disambiguated_publishers[outlet]['entity'], entity_type='ORG'))
                resource = kg_utils.label2custom_resource(label, self.namespaces['custom_resource'])
            disambiguated_publishers[outlet]['label'] = label
            disambiguated_publishers[outlet]['resource'] = resource

        target_indices = [self._node_id2news_idx(data, node_id, self.node_prefix['news'])[0] for node_id in source_nodes]
        target_outlets = data.loc[target_indices][self.data2include['publisher']].tolist()

        target_nodes = [disambiguated_publishers[outlet]['resource'] for outlet in target_outlets]
        target_labels = [disambiguated_publishers[outlet]['label'] for outlet in target_outlets]

        self._add_simple_node_relation(source_nodes=source_nodes, target_nodes=target_nodes, property_type=self.predicates['publisher'])
        self.set_nodes_labels(list(zip(target_nodes, target_labels)))

        return disambiguated_publishers

    def _get_overlap_ratio(self, sequence1: str, sequence2: str) -> float:
        """ Returns a measure of sequences' similarity between [0, 1], given by the ratio of number of matches to the total number of elements in both sequences. """ 
        s = SequenceMatcher(None, sequence1, sequence2) 
        return s.ratio()

    def _get_related_articles(self, data: pd.DataFrame) -> List[Tuple[int, int]]:
        updated_articles = data[data.duplicated(subset='title', keep=False)==True]
        grouped_updated_articles = updated_articles.groupby('title').apply(lambda x: x.index.to_list())
        updated_articles_idx = grouped_updated_articles.values.tolist()
 
        overlapping_articles_idx = list()
        for idx_pair in updated_articles_idx:
            overlap_ratio = self._get_overlap_ratio(
                    data.loc[idx_pair[0]][self.data2include['body']], 
                    data.loc[idx_pair[1]][self.data2include['body']]
                    )
            if overlap_ratio >= self.related_news_threshold:
                # Consider only articles with an overlap ratio higher than the threshold
                sample = updated_articles.loc[idx_pair]

                if sample.loc[idx_pair[0]][self.data2include['date_published']] != sample.loc[idx_pair[1]][self.data2include['date_published']]:
                    # Order by creation date
                    older_article_idx = sample.loc[sample[self.data2include['date_published']]==sample[self.data2include['date_published']].min()].index.values.astype(int)[0]
                    newer_article_idx = sample.loc[sample[self.data2include['date_published']]==sample[self.data2include['date_published']].max()].index.values.astype(int)[0]
                elif sample.loc[idx_pair[0]][self.data2include['date_published']] != sample.loc[idx_pair[1]][self.data2include['date_published']]:
                    # Order by last modification date
                    older_article_idx = sample.loc[sample[self.data2include['date_modified']]==sample[self.data2include['date_modified']].min()].index.values.astype(int)[0]
                    newer_article_idx = sample.loc[sample[self.data2include['date_modified']]==sample[self.data2include['date_modified']].max()].index.values.astype(int)[0]
                else: 
                    # Order by artile length
                    older_article_idx = sample.loc[sample[self.data2include['body']].str.len().idxmin()].name
                    newer_article_idx = sample.loc[sample[self.data2include['body']].str.len().idxmax()].name
                overlapping_articles_idx.append([newer_article_idx, older_article_idx])

        overlapping_articles_ids = [data.loc[idx][self.data2include['id']].values.tolist() for idx in overlapping_articles_idx]

        return overlapping_articles_ids

    def _remove_infrequent_unlinked_resources(self):
        log.info('Removing infrequent unlinked nodes from the graph.')
        unlinked_resources = self.get_unlinked_resources()
        infrequent_nodes = [node for node in unlinked_resources if self.degree(node) < self.min_unlinked_resource_freq]
        log.info(f'Found {len(infrequent_nodes)} infrequent nodes.')

        self._remove_nodes(infrequent_nodes)

    def _remove_sink_entities(self):
        log.info('Removing sink entities from the graph.')
        wikidata_resources = self.get_wikidata_resources()
        sink_entities = [node for node in wikidata_resources if self.degree(node) < self.min_sink_entities_freq]
        log.info(f'Found {len(sink_entities)} sink entities.')

        self._remove_nodes(sink_entities)

