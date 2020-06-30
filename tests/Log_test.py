from ivtools import settings

log = settings.log

log.debug("This is a debug message.")
log.info("This is an info message.")
log.warning('This is a warning.')
log.error('This is an error.')
log.testlevel('This a level created by me.')
log.critical("This is critical!")
