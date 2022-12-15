import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(version_base="1.2", config_path="configs/", config_name="pipeline.yaml")
def main(cfg: DictConfig):

    import traceback

    from src.utils import utils
    from src.pipeline import run

    # Applied optional utilities
    utils.extras(cfg)

    # Run the pipeline
    try:
        return run(cfg)
    except Exception:
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    main()
