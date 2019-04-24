import logging


def custom_logger(name=__name__, level=logging.DEBUG):
    """
    Custom logging message formatting.
    :param name: (str) custom logger name
    :param level: (int) logging level, default is DEBUG
    :return: (class) logging.Logger
    """

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-8s %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # Handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Custom logger
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
