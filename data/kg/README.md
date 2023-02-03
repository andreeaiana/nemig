# KG Triple Files

NeMigKG is serialized as [gzipped](https://www.gzip.org/) files in [N-Triples](https://www.w3.org/TR/n-triples/) format, and available under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

The samples available here contain the complete NeMigKG without the bodies of the news. The complete data dump can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7442425).

| File | Description | 
|------|-------------|
| nemig_${language}_ ${graph_type}-metadata.nt.bz2    |    Metadata about the dataset, described using void vocabulary.      
| nemig_${language}_ ${graph_type}-instances_types.nt.bz2    |    Class definitions of news and event instances.
| nemig_${language}_ ${graph_type}-instances_labels.nt.bz2    |    Labels of instances.
| nemig_${language}_ ${graph_type}-instances_related.nt.bz2    |    Relations between news instances based on one another.
| nemig_${language}_ ${graph_type}-instances_metadata_literals.nt.bz2    |    Relations between news instances and metadata literals (e.g. URL, publishing date, modification date, sentiment label, political orientation of news outlets).
| nemig_${language}_ ${graph_type}-instances_content_mapping.nt.bz2    |    Mapping of news instances to content instances (e.g. title, abstract, body).
| nemig_${language}_ ${graph_type}-instances_topic_mapping.nt.bz2    |    Mapping of news instances to sub-topic instances.
| nemig_${language}_ ${graph_type}-instances_content_literals.nt.bz2    |    Relations between content instances and corresponding literals (e.g. text of title, abstract, body).
| nemig_${language}_ ${graph_type}-instances_metadata_resources.nt.bz2    |    Relations between news or sub-topic instances and entities extracted from metadata (i.e. publishers, authors, keywords).
| nemig_${language}_ ${graph_type}-instances_event_mapping.nt.bz2    |    Mapping of news instances to event instances.
| nemig_${language}_ ${graph_type}-event_resources.nt.bz2    |    Relations between event instances and entities extracted from the text of the news (i.e. actors, places, mentions).
| nemig_${language}_ ${graph_type}-resources_provenance.nt.bz2    |    Provenance information about the entities extracted from the text of the news (e.g. title, abstract, body).
| nemig_${language}_ ${graph_type}-wiki_resources.nt.bz2    |    Relations between Wikidata entities from news and their k-hop entity neighbors from Wikidata.
