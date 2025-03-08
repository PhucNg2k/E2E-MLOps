from cnnClassifier import logger

from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline


def run_stage(StageName, pipepline):
    try:
        logger.info(f">>>>>> stage {StageName} started <<<<<<")
        obj = pipepline()
        obj.main()
        logger.info(f">>>>>> stage {StageName} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    
    STAGE_NAME = "Data Ingestion stage"
    run_stage(STAGE_NAME, DataIngestionTrainingPipeline)
    
    STAGE_NAME = "Prepare Base Model stage"
    run_stage(STAGE_NAME, PrepareBaseModelTrainingPipeline)
    