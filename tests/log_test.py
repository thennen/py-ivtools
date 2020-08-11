import ivtools
import logging

if __name__ == '__main__':
    loggers = ['instruments', 'io', 'plots', 'analyze', 'interactive', None]
    for logger in loggers:
        log = logging.getLogger(logger)

        log.debug(f"This is a 'debug' message from {logger}")
        log.info(f"This is an 'info' message from {logger}")
        log.warning(f"This is a 'warning' message from {logger}")
        log.error(f"This is an 'error' message from {logger}")
        log.critical(f"This is a 'critical' message from {logger}")

    log = logging.getLogger('io')
    log.info("This is a very complicated message:\nMira que movidas hago:\n\tHola Hola Hola\n\tAdios")
