from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>>> Starting Data Transformation stage <<<<")

            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.convert()

            logger.info("Data Transformation completed successfully!")
            logger.info(">>>> Data Transformation stage finished <<<<\n")

        except Exception as e:
            logger.exception(f"Error occurred during Data Transformation: {e}")
            raise e
