from .p2pnet import build as b1
from .p2pnet2 import build as b2

# build the P2PNet model
# set training to 'True' during training
def build_model(args, adapter=False, training=False):
    if adapter:
        return b2(args, training)
    return b1(args, training)