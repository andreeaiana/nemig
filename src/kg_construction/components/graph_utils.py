def is_custom_resource(item: str, namespace_custom_resource: str) -> bool:
    return type(item)==str and item.startswith(namespace_custom_resource)


def id2custom_resource(node_id: str, namespace_custom_resource: str) -> str:
    return namespace_custom_resource + node_id


def custom_resource2id(custom_resource: str, namespace_custom_resource: str) -> str:
    return custom_resource[len(namespace_custom_resource):]


def label2custom_resource(label: str, namespace_custom_resource: str) -> str:
    return namespace_custom_resource + label


def custom_resource2label(custom_resource: str, namespace_custom_resource: str) -> str:
    return custom_resource[len(namespace_custom_resource):]


def is_custom_property(item: str, namespace_custom_property: str) -> bool:
    return item.startswith(namespace_custom_property)


def label2custom_property(label: str, namespace_custom_property: str) -> str:
    return namespace_custom_property + label


def custom_property2label(custom_property: str, namespace_custom_property: str) -> str:
    return custom_property[len(namespace_custom_property):]


def qid2wikidata_resource(qid: str, namespace_wikidata_resource: str) -> str:
    return namespace_wikidata_resource + qid


def wikidata_resource2qid(wikidata_resource: str, namespace_wikidata_resource: str) -> str:
    return wikidata_resource.split(namespace_wikidata_resource)[-1]


def is_wikidata_resource(item: str, namespace_wikidata_resource: str) -> bool:
    return type(item)==str and item.startswith(namespace_wikidata_resource)


def pid2wikidata_property(pid: str, namespace_wikidata_property) -> str:
    return namespace_wikidata_property + pid

