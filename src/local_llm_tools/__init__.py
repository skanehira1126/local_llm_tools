from logging import Formatter
from logging import StreamHandler
from logging import getLogger


logger = getLogger(__name__)
fmt = Formatter(
    "[%(levelname)s] %(name)s %(asctime)s - %(filename)s: %(lineno)d: %(message)s",
)
sh = StreamHandler()
sh.setLevel("INFO")
sh.setFormatter(fmt)
logger.addHandler(sh)
logger.setLevel("INFO")
