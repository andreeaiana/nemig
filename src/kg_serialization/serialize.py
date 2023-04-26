""" Functionality to serialize the individual parts of the NeMigKG. """

from typing import List, Dict, Any

import bz2
from src.kg_construction.news_kg import NewsKG
import src.kg_serialization.utils as serial_utils
import src.kg_construction.components.graph_utils as kg_utils
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class TriplesSerializer():
    def __init__(
            self,
            graph_name: str,
            creation_date: str,
            description: str, 
            namespaces: Dict[str, str],
            predicates: Dict[str, str],
            type_resource: str,
            postfixes: Dict[str, str],
            resource_encoded_chars: List[str],
            literal_encoded_chars: List[str],
            triples_to_serialize: List[str],
            triples_files: Dict[str, str],
            void_resource: str,
            ) -> None:

        self.graph_name = graph_name
        self.creation_date = creation_date
        self.namespaces = namespaces
        self.predicates = predicates
        self.type_resource = type_resource
        self.postfixes = postfixes
        self.resource_encoded_chars = resource_encoded_chars
        self.literal_encoded_chars = literal_encoded_chars
        self.triples_to_serialize = triples_to_serialize
        self.triples_files = triples_files
        self.void_resource = void_resource
        self.description = description

        assert all([triple in self.triples_files.keys() for triple in self.triples_to_serialize])

    def serialize_graph(self, news_kg: NewsKG) -> None:
        self.graph = news_kg

        if 'metadata' in self.triples_to_serialize:
            log.info('Serializing metadata.')
            lines = self._get_lines_metadata()
            file = self.triples_files['metadata']
            self._write_lines_to_file(lines, file)

        if 'instances_types' in self.triples_to_serialize:
            log.info('Serializing types of instances.')
            lines = self._get_lines_instances_types()
            file = self.triples_files['instances_types']
            self._write_lines_to_file(lines, file)

        if 'instances_labels' in self.triples_to_serialize:
            log.info('Serializing labels of instances.')
            lines = self._get_lines_instances_labels()
            file = self.triples_files['instances_labels']
            self._write_lines_to_file(lines, file)

        if 'instances_related' in self.triples_to_serialize:
            log.info('Serializing related instances.')
            properties = [
                    self.predicates['is_based_on']
                    ]
            lines = self._get_lines_instances_relations(properties)
            file = self.triples_files['instances_related']
            self._write_lines_to_file(lines, file)

        if 'instances_metadata_literals' in self.triples_to_serialize:
            log.info('Serializing metadata literals.')
            properties = [
                    self.predicates['url'],
                    self.predicates['date_published'],
                    self.predicates['date_modified'],
                    ]
            lines = self._get_lines_instances_relations(properties)
            file = self.triples_files['instances_metadata_literals']
            self._write_lines_to_file(lines, file)

        if 'instances_content_mapping' in self.triples_to_serialize:
            log.info('Serializing mapping of news to content resources.')
            lines = self._get_lines_instances_content_mapping()
            file = self.triples_files['instances_content_mapping']
            self._write_lines_to_file(lines, file)

        if 'instances_topic_mapping' in self.triples_to_serialize:
            log.info('Serializing mapping of news to topic resources.')
            lines = self._get_lines_instances_topic_mapping()
            file = self.triples_files['instances_topic_mapping']
            self._write_lines_to_file(lines, file)

        if 'instances_sentiment_mapping' in self.triples_to_serialize:
            log.info('Serializing mapping of news to sentiment resources.')
            lines = self._get_lines_instances_sentiment_mapping()
            file = self.triples_files['instances_sentiment_mapping']
            self._write_lines_to_file(lines, file)

        if 'instances_political_orientation_mapping' in self.triples_to_serialize:
            log.info('Serializing mapping of news outlets to political orientation resources.')
            lines = self._get_lines_instances_political_orientation_mapping()
            file = self.triples_files['instances_political_orientation_mapping']
            self._write_lines_to_file(lines, file)

        if 'instances_content_literals' in self.triples_to_serialize:
            log.info('Serializing content literals.')
            properties = [
                    self.predicates['headline'],
                    self.predicates['abstract'],
                    self.predicates['article_body']
                    ]
            lines = self._get_lines_instances_relations(properties)
            file = self.triples_files['instances_content_literals']
            self._write_lines_to_file(lines, file)

        if 'instances_sentiment_polorient_literals' in self.triples_to_serialize:
            log.info('Serializing sentiment literals.')
            properties = [
                    self.predicates['description']
                    ]
            lines = self._get_lines_instances_relations(properties)
            file = self.triples_files['instances_sentiment_polorient_literals']
            self._write_lines_to_file(lines, file)

        if 'instances_metadata_resources' in self.triples_to_serialize:
            log.info('Serializing metadata resources.')
            properties = [
                    self.predicates['publisher'],
                    self.predicates['author'],
                    self.predicates['keywords']
                    ]
            lines = self._get_lines_instances_relations(properties)
            file = self.triples_files['instances_metadata_resources']
            self._write_lines_to_file(lines, file)

        if 'instances_event_mapping' in self.triples_to_serialize:
            log.info('Serializing news mappings to event resources.')
            lines = self._get_lines_instances_event_mapping()
            file = self.triples_files['instances_event_mapping']
            self._write_lines_to_file(lines, file)

        if 'instances_event_resources' in self.triples_to_serialize:
            log.info('Serializing event resources.')
            properties = [
                    self.predicates['has_actor'],
                    self.predicates['has_place'],
                    self.predicates['mentions']
                    ]
            lines = self._get_lines_instances_relations(properties)
            file = self.triples_files['instances_event_resources']
            self._write_lines_to_file(lines, file)

        if 'instances_resources_provenance' in self.triples_files:
            log.info('Serializing provenance of resources.')
            properties = [
                    self.predicates['is_referenced_by']
                    ]
            lines = self._get_lines_instances_relations(properties)
            file = self.triples_files['instances_resources_provenance']
            self._write_lines_to_file(lines, file)

        if 'instances_wiki_resources' in self.triples_to_serialize:
            log.info('Serializing Wikidata resources and relations.')
            properties = [prop for prop in self.graph.get_all_properties() if 'wiki' in prop]
            lines = self._get_lines_instances_relations(properties)
            file = self.triples_files['instances_wiki_resources']
            self._write_lines_to_file(lines, file)

    def _as_literal_triple(self, subj:str, pred: str, obj: Any) -> str:
        return serial_utils.as_literal_triple(
                subj=subj,
                pred=pred,
                obj=obj,
                literal_encoded_chars=self.literal_encoded_chars,
                type_resource=self.type_resource,
                postfixes=self.postfixes,
                resource_encoded_chars=self.resource_encoded_chars
                )

    def _as_object_triple(self, subj: str, pred: str, obj: str) -> str:
        return serial_utils.as_object_triple(
                subj=subj,
                pred=pred,
                obj=obj,
                type_resource=self.type_resource,
                postfixes=self.postfixes,
                resource_encoded_chars=self.resource_encoded_chars
                )

    def _get_lines_instances_relations(self, properties: List[str]) -> List[str]:
        """ Serialize resource facts for the given properties. """
        lines_instance_relations = list()
        instances_relations = set()

        for prop in properties:
            instances_relations.update({(res, prop, val) for (res, val) in self.graph.get_nodes_for_property(prop)})

        for s, p, o in instances_relations:
            if kg_utils.is_custom_resource(o, self.namespaces['custom_resource']) or kg_utils.is_wikidata_resource(o, self.namespaces['wikidata_resource']):
                lines_instance_relations.append(self._as_object_triple(s, p, o))
            else:
                lines_instance_relations.append(self._as_literal_triple(s, p, o))
        return lines_instance_relations    

    def _get_lines_instances_labels(self) -> List[str]:
        """ Serialize the labels of instances. """
        resources = list(self.graph.get_unlinked_resources().union(self.graph.get_wikidata_resources()))
        labels = [self.graph.get_node_label(resource) for resource in resources]
        labels = [label if not label == None else 'No label defined' for label in labels]

        instances_labels = list()
        for instance, label in zip(resources, labels):
            instances_labels.append(self._as_literal_triple(instance, self.predicates['label'], label))

        return instances_labels

    def _get_lines_instances_types(self) -> List[str]:
        """ Serialize types of resources. """
        list_instances_types = list()

        for res, res_type in self.graph.get_nodes_for_property(self.predicates['type']):
            list_instances_types.append(self._as_object_triple(res, self.predicates['type'], res_type))
        return list_instances_types

    def _get_lines_instances_content_mapping(self) -> List[str]:
        """ Serialize content mapping for news article resources. """
        content_nodes = self.graph.get_content_nodes()

        instance_content_mappings = list()
        for res, content in self.graph.get_nodes_for_property(self.predicates['has_part']):
            if content in content_nodes:
                instance_content_mappings.append(self._as_object_triple(res, self.predicates['has_part'], content))
        
        return instance_content_mappings

    def _get_lines_instances_topic_mapping(self) -> List[str]:
        """ Serialize topic mapping for news article resources. """
        topic_nodes = self.graph.get_topic_nodes()

        instance_topic_mappings = list()
        for res, topic in self.graph.get_nodes_for_property(self.predicates['about']):
            if topic in topic_nodes:
                instance_topic_mappings.append(self._as_object_triple(res, self.predicates['about'], topic))

        return instance_topic_mappings

    def _get_lines_instances_sentiment_mapping(self) -> List[str]:
        """ Serialize sentiment mapping for news article resources. """
        sentiment_nodes = self.graph.get_sentiment_nodes()

        instance_sentiment_mappings = list()
        for res, sentiment in self.graph.get_nodes_for_property(self.predicates['sentiment']):
            if sentiment in sentiment_nodes:
                instance_sentiment_mappings.append(self._as_object_triple(res, self.predicates['sentiment'], sentiment))

        return instance_sentiment_mappings

    def _get_lines_instances_political_orientation_mapping(self) -> List[str]:
        """ Serialize political orientation mapping for news outlet resources. """
        political_orientation_nodes = self.graph.get_pol_orient_nodes()

        instance_pol_orient_mappings = list()
        for res, pol_orient in self.graph.get_nodes_for_property(self.predicates['political_orientation']):
            if pol_orient in political_orientation_nodes:
                instance_pol_orient_mappings.append(self._as_object_triple(res, self.predicates['political_orientation'], pol_orient))

        return instance_pol_orient_mappings
    
    def _get_lines_instances_event_mapping(self) -> List[str]:
        """ Serialize event mapping for news article resources. """
        event_nodes = self.graph.get_event_nodes()

        instance_event_mappings = list()
        for res, event in self.graph.get_nodes_for_property(self.predicates['about']):
            if event in event_nodes:
                instance_event_mappings.append(self._as_object_triple(res, self.predicates['about'], event))

        return instance_event_mappings

    def _write_lines_to_file(self, lines: List, filepath: str) -> None:
        with bz2.open(filepath, mode='wt') as f:
            f.writelines(lines)


    def _get_lines_metadata(self) -> List[str]:
        """ Serialize metadata. """
        log.info('GeNeG: Serializing metadata')
        
        entity_count = len(self.graph.get_custom_resources()) + len(self.graph.get_wikidata_resources())
        property_count = len(self.graph.get_all_properties())
        return [
                self._as_object_triple(self.void_resource, self.predicates['type'], 'http://rdfs.org/ns/void#Dataset'),
                self._as_literal_triple(self.void_resource, 'http://purl.org/dc/elements/1.1/title', self.graph_name),
                self._as_literal_triple(self.void_resource, self.predicates['label'], self.graph_name),
                self._as_literal_triple(self.void_resource, 'http://purl.org/dc/elements/1.1/description', self.description),
                self._as_object_triple(self.void_resource, 'http://purl.org/dc/terms/license', 'http://creativecommons.org/licenses/by-nc-sa/4.0/'),
                self._as_literal_triple(self.void_resource, 'http://purl.org/dc/terms/creator', 'Andreea Iana, Heiko Paulheim'),
                self._as_literal_triple(self.void_resource, 'http://purl.org/dc/terms/created', self.creation_date),
                self._as_literal_triple(self.void_resource, 'http://purl.org/dc/terms/publisher', 'Andreea Iana'),
                self._as_literal_triple(self.void_resource, 'http://rdfs.org/ns/void#uriSpace', self.namespaces['custom_resource']),
                self._as_literal_triple(self.void_resource, 'http://rdfs.org/ns/void#entities', entity_count),
                self._as_literal_triple(self.void_resource, 'http://rdfs.org/ns/void#properties', property_count),
       ]

