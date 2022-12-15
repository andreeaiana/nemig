from typing import Dict

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.utils import pylogger
from src.dataset.dataset import NewsDataset
from src.sentiment_classification.model import SentimentClassificationModel
from src.named_entity_recognition.model import NERModel
from src.named_entity_linking.model import NELModel
from src.entity_filtering.model import EntityFilter
from src.kg_construction.news_kg import NewsKG
from src.kg_serialization.serialize import TriplesSerializer

log = pylogger.get_pylogger(__name__)


def run(cfg: DictConfig) -> Dict:
    """
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating dataset <{cfg.dataset._target_}>")
    dataset: NewsDataset = hydra.utils.instantiate(cfg.dataset)

    log.info(f"Instantiating sentiment classification model <{cfg.sentiment_classification._target_}>")
    sentiment_classification_model: SentimentClassificationModel = hydra.utils.instantiate(cfg.sentiment_classification, ds_conf=cfg.dataset, _recursive_=False)

    log.info(f"Instantiating named entity recognition model <{cfg.named_entity_recognition._target_}>")
    ner_model: NERModel = hydra.utils.instantiate(cfg.named_entity_recognition, ds_conf=cfg.dataset, _recursive_=False)
    
    log.info(f"Instantiating named entity linking model <{cfg.named_entity_linking._target_}>")
    nel_model: NELModel = hydra.utils.instantiate(cfg.named_entity_linking)

    log.info(f"Instantiating entity filtering model <{cfg.entity_filtering._target_}>")
    entity_filter: EntityFilter = hydra.utils.instantiate(cfg.entity_filtering)
    
    log.info(f"Instantiating news knowledge graph <{cfg.kg_construction._target_}>")
    news_kg: NewsKG = hydra.utils.instantiate(cfg.kg_construction)

    log.info(f"Instantiating triples serializer <{cfg.kg_serialization._target_}>")
    triples_serializer: TriplesSerializer = hydra.utils.instantiate(cfg.kg_serialization)

    object_dict = {
        "config": cfg,
        "dataset": dataset,
        "sentiment_classification": sentiment_classification_model,
        "named_entity_recognition": ner_model,
        "named_entity_linking": nel_model,
        'entity_filtering': entity_filter,
        "kg_contruction": news_kg,
        "kg_serialization": triples_serializer
    }

    if cfg.get("run"):
        log.info("Starting running the pipeline!")
        
        log.info("Preparing the dataset.")
        dataset.load_data()
        
        log.info("Running sentiment analysis")
        sentiment_classification_model.classify_articles()
        
        log.info("Running named entity recognition.")
        ner_model.extract_entities()

        log.info("Running named entity linking.")
        nel_model.link_entities()

        log.info("Running entity filtering.")
        entity_filter.filter_entities()
    
        log.info("Running knowledge graph construction.")
        graph = news_kg.get_graph()

        log.info("Serializing knowledge graph as triples.")
        triples_serializer.serialize_graph(news_kg)

        log.info(f"Finished")

    return object_dict 

