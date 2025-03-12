import os
import sys
import traceback
from werkzeug.debug import get_current_traceback

def enable_file_logging():
    """Set up logging to a file for debugging purposes"""
    import logging
    log_file = 'dfm_analysis_debug.log'
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('dfm_analysis')
    logger.info("Starting application with debug logging")
    return logger

logger = enable_file_logging()

def log_exception(e):
    """Log an exception with full traceback"""
    exc_info = sys.exc_info()
    tb = traceback.format_exception(*exc_info)
    logger.error("Exception occurred:\n%s", ''.join(tb))
    
    # Get detailed Werkzeug traceback
    werkzeug_tb = get_current_traceback(skip=1, show_hidden_frames=True,
                                        ignore_system_exceptions=False)
    logger.error("Werkzeug traceback:\n%s", werkzeug_tb.plaintext)

# Usage example:
# try:
#     something_that_might_fail()
# except Exception as e:
#     log_exception(e)
#     raise