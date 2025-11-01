import os
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            # Path where data was extracted
            data_path = os.path.join("artifacts", "data_ingestion", "samsum_dataset")
            
            # List all files/folders in that directory
            all_files = os.listdir(data_path)
            logger.info(f"Files found in dataset: {all_files}")

            # Find missing files
            missing_files = [file for file in self.config.ALL_REQUIRED_FILES if file not in all_files]

            # Determine validation status
            validation_status = len(missing_files) == 0

            # Write status to file
            with open(self.config.STATUS_FILE, 'w') as f:
                if validation_status:
                    f.write(f"Validation Status: SUCCESS \nAll required files are present.")
                    logger.info("All required files are present.")
                else:
                    f.write(f"Validation Status: FAILED \nMissing files: {missing_files}")
                    logger.error(f"Missing files: {missing_files}")

            return validation_status

        except Exception as e:
            logger.exception(e)
            raise e