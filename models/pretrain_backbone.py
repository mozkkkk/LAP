from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone




# the defenition of the P2PNet model
class P2PNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d([2,2])
        self.fc = nn.Linear(2048,self.num_classes)


    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        feat=self.pool(features[3])
        classification = self.fc(feat.flatten(1,3))
        output_class = classification

        return output_class


# create the P2PNet model
def build(args, training):
    # treats persons as a single class
    num_classes = args.num_classes

    backbone = build_backbone(args)
    model = P2PNet(backbone, num_classes)
    if not training:
        return model

    criterion = nn.CrossEntropyLoss()

    return model, criterion
