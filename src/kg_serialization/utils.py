""" Serialization of statements into RDF triples """

from typing import List, Dict, Any

import urllib.parse


def as_literal_triple(
        subj: str, 
        pred: str, 
        obj: Any, 
        literal_encoded_chars: List[str], 
        type_resource: str, 
        postfixes: Dict[str, str], 
        resource_encoded_chars: List[str]
        ) -> str:
    """ Serialize a triple as a literal triple. """
    obj_type = type(obj)
    if obj_type == str:
        obj = _encode_literal_string(obj, literal_encoded_chars)
    return _as_triple(subj, pred, obj, 
                      obj_type=obj_type, 
                      type_resource=type_resource, 
                      postfixes=postfixes, 
                      resource_encoded_chars=resource_encoded_chars
                      )


def as_object_triple(
        subj: str, 
        pred: str, 
        obj: str, 
        type_resource: str, 
        postfixes: Dict[str, str], 
        resource_encoded_chars: List[str]
        ) -> str:
    """ Serialize a triple as an object triple. """
    return _as_triple(subj, pred, obj, 
                      obj_type=type_resource, 
                      type_resource=type_resource, 
                      postfixes=postfixes, 
                      resource_encoded_chars=resource_encoded_chars
                      )


def _as_triple(
        subj: str, 
        pred: str, 
        obj: str, 
        obj_type: Any, 
        type_resource: str, 
        postfixes: Dict[str, str], 
        resource_encoded_chars: List[str]
        ) -> str:
    if obj_type == type_resource:
        obj_as_string = _resource_to_string(obj, resource_encoded_chars=resource_encoded_chars)
    else:
        obj_as_string = f'"{obj}"'
        if obj_type in postfixes.keys():
            obj_as_string += f'^^{_resource_to_string(postfixes[obj_type], resource_encoded_chars)}'
    return f'{_resource_to_string(subj, resource_encoded_chars)} {_resource_to_string(pred, resource_encoded_chars)} {obj_as_string} .\n'


def _resource_to_string(resource: str, resource_encoded_chars: List[str]) -> str:
    for c in resource_encoded_chars:
        resource = resource.replace(c, urllib.parse.quote_plus(c))
    return resource if resource.startswith('_:') else f'<{resource}>'


def _encode_literal_string(literal: str, literal_encoded_chars: List[str]) -> str:
    for c in literal_encoded_chars:
        literal = literal.replace(c, f'\\{c}')
    return literal

