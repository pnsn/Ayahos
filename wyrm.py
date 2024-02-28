import logging

module_logger = logging.getLogger(__name__)

class Wyrm:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of Wyrm')

    def pulse(self, x):
        self.logger.debug('initiating pulse')
        y = x
        self.logger.debug('concluding pulse')
        return y
    
def wyrm_module_function():
    module_logger.info('received a call to "wyrm_module_function"')