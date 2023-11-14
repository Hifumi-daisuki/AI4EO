from trainner.cityscapes import cityscapes_bisenetv2_single_gpu_trainner as single_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

LOG = init_logger.get_logger('train_bisenetv2_cityscapes')
CFG = parse_config_utils.cityscapes_cfg_v2


def train_model():
    LOG.info('Using single gpu trainner ...')
    worker = single_gpu_trainner.BiseNetV2CityScapesTrainer()

    worker.train()
    return

if __name__ == '__main__':

    train_model()
