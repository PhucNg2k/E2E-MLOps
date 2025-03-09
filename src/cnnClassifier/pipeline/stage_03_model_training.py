from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger

STAGE_NAME = "Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        configManager = ConfigurationManager()
        training_config = configManager.get_training_config()
        training = Training(config=training_config)

        

        training.get_base_model()
        training.train_val_DataGenerator()
        training.train()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


