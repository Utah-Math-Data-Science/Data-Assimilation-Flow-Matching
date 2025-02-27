import pytest
import hydra

from conf import conf
from dafm import utils


@pytest.fixture
def engine():
    engine = conf.sa.create_engine('sqlite+pysqlite:///:memory:')
    conf.orm.create_all(engine)
    return engine


def init_hydra_cfg(config_name, overrides, config_dir=str(utils.DIR_ROOT/'conf')):
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(config_name=config_name, overrides=overrides)
