import ivtools
import logging

log = logging.getLogger('my_logger')

log.debug("This is a 'debug' message")
log.info("This is an 'info' message")
log.warning("This is a 'warning' message")
log.error("This is an 'error' message")
log.critical("This is a 'critical' message")
log.instruments("This is an 'instruments' message")
log.io("This is an 'io' message")
log.plots("This is a 'plot' message")
log.analysis("This is an 'analysis' message")
log.interactive("This is an 'interactive' message")