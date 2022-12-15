# NeMig
NeMig represents a bilingual news collection and knowledge graphs on the topic of migration. The news corpora in German and English were collected from online media outlets from Germany and the US, respectively. NeMIg contains rich textual and metadata information, sub-topics and sentiment annotations, as well as named entities extracted from the articles' content and metadata and linked to Wikidata. The corresponding knowledge graphs built from each corpus are expanded with up to two-hop neighbors from Wikidata of the initial set of linked entities.

## Features
NeMig comes in four flavors, for both the German, and the English corpora:

- **Base NeMig**: contains literals and entities from the corresponding annotated news corpus;
- **Entities NeMig**: derived from the Base NeMIg by removing all literal nodes, it contains only resource nodes;
- **Enriched Entities NeMig**: derived from the Entities NeMig by enriching it with up to two-hop neighbors from Wikidata, it contains only resource nodes and Wikidata triples;
- **Complete NeMig**: the combination of the Base and Enriched Entities NeMig, it contains both literals and resources.

## Project Structure

The directory structure of new project looks like this:

```
├── configs                             <- Hydra configuration files
│   ├── dataset                            <- Dataset configs
│   ├── entity_filtering                   <- Entity filtering configs
│   ├── experiment                         <- Experiment configs
│   ├── hydra                              <- Hydra configs
│   ├── kg_construction                    <- Knowledge graph construction configs
│   ├── kg_serialization                   <- Koowledge graph serialization configs
│   ├── named_entity_linking               <- Named entity linking configs
│   ├── named_entity_recognition           <- Named entity recognition configs
│   ├── sentiment_classification           <- Sentiment classification configs
│   │
│   ├── pipeline.yaml                   <- Main config for the pipeline
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra loggers
│
├── notebooks              <- Jupyter notebooks 
│
├── scripts                <- Shell scripts
│
├── src                             <- Source code
│   ├── dataset                            <- Dataset creation and processing
│   ├── entity_filtering                   <- Entity filtering model
│   ├── kg_construction                    <- Knowledge graph construction model
│   ├── kg_serialization                   <- Koowledge graph serialization model
│   ├── named_entity_linking               <- Named entity linking model
│   ├── named_entity_recognition           <- Named entity recognition model
│   ├── sentiment_classification           <- Sentiment classification model
│   ├── utils                              <- Utility scripts
│   │
│   └── pipeine.py                 <- Run pipeline
│
├── .gitignore                <- List of files ignored by git
├── requirements.txt          <- File for installing python dependencies
└── README.md
```

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/andreeaiana/nemig
cd nemig

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install requirements
pip install -r requirements.txt
```

Download the mGENRE model as described in [mGENRE](https://github.com/facebookresearch/GENRE/tree/main/examples_mgenre) needed for running the entity linking model.

Run pipeline with default configuration

```bash
python src/main.py language=$

```

Run pipeline with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/main.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/main.py language='de' kg_construction.k_hop=1
```

Run the [Subtopic Modelling notebook](notebooks/1.0-subtopic-modeling-en.ipynb) to extract sub-topics from the data and integrate the results in the pipeline.

The chosen version of NeMig will be constructed and cached in the [cache folder](data/cache/). NeMigKG is serialized in N-Triple format, and the resulting files are placed in the [kg folder](data/kg/).

## Data
A sample of the annotated news corpora used to construct the knowledge graphs are available in the [cache folder](data/cache). Due to copyright policies, this sample does not contain the body of the articles.

The anonymized user data and the associated subset of the German corpora are available in the [news-recommendation folder](data/news_recommendation/).

A full version of the news corpus is available [upon request](mailto:andreea.iana@uni-mannheim.de).

## Results
[NeMigKG](https://doi.org/10.5281/zenodo.7442425) is hosted on [Zenodo](https://zenodo.org/). All files are [gzipped](https://www.gzip.org/) and in [N-Triples format](https://www.w3.org/TR/n-triples/). 


## License
The code is licensed under the MIT License. The data files are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

