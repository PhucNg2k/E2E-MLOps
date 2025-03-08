from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


STAGE_NAME = "Prepare Base Model stage"


class PrepareBaseModelTrainingPipeline:

    def __init__(self):
        pass
    
    def main(self):
        configManager = ConfigurationManager()
        prepare_base_model_config = configManager.get_prepre_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


