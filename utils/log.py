import logging


def thread_log():
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(message)s]"
    )
