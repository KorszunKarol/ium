"""
Logging configuration for the API

Author: Deployment Team
Created: 2025-05-31
"""

import logging
import logging.config
import os
from datetime import datetime

def setup_logging(log_level: str = "INFO"):
   """Setup logging configuration"""


   log_dir = "logs"
   os.makedirs(log_dir, exist_ok=True)


   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   log_file = os.path.join(log_dir, f"nocarz_api_{timestamp}.log")

   logging_config = {
       "version": 1,
       "disable_existing_loggers": False,
       "formatters": {
           "standard": {
               "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
               "datefmt": "%Y-%m-%d %H:%M:%S"
           },
           "detailed": {
               "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
               "datefmt": "%Y-%m-%d %H:%M:%S"
           }
       },
       "handlers": {
           "console": {
               "level": log_level,
               "class": "logging.StreamHandler",
               "formatter": "standard"
           },
           "file": {
               "level": "DEBUG",
               "class": "logging.FileHandler",
               "filename": log_file,
               "formatter": "detailed"
           }
       },
       "loggers": {
           "": {
               "handlers": ["console", "file"],
               "level": "DEBUG",
               "propagate": False
           },
           "uvicorn": {
               "handlers": ["console", "file"],
               "level": "INFO",
               "propagate": False
           },
           "uvicorn.access": {
               "handlers": ["console", "file"],
               "level": "INFO",
               "propagate": False
           }
       }
   }

   logging.config.dictConfig(logging_config)


   logger = logging.getLogger(__name__)
   logger.info(f"Logging setup completed. Log file: {log_file}")

   return log_file
