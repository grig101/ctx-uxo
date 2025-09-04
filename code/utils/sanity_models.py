import torch
import sys
sys.path.append("../")
from model.backbone import ResNetBackbone
from model.region_proposal import RegionProposal
from model.head import Head

# Custom function
# If you want to build a custom FASTER RCNN for UXO detection!
def run_sanity_checks(batch_size=2, img_size=640, num_classes=2, seed=23, logger = None):
    torch.manual_seed(seed)
    all_ok = True

    try:
        backbone = ResNetBackbone()
        backbone.eval()
        dummy_img = torch.randn(batch_size, 3, img_size, img_size)
        features = backbone(dummy_img)
        if logger:
            logger.info(f"Backbone output type: {type(features)}")
            if isinstance(features, dict):
                for k, v in features.items():
                    logger.info(f"  Feature map {k}: {v.shape}")
            else:
                logger.info(f"Backbone output shape: {features.shape}")
            logger.warning(f"Backbone forward OK!\n")
    except Exception as e:
        if logger:
            logger.error(f"Backbone error: {e}")
        all_ok = False

    try:
        rpn = RegionProposal(in_channels=256)
        rpn.eval()
        batch_dims = [(img_size, img_size)] * batch_size
        proposals, rpn_losses = rpn(dummy_img, features, batch_dims, targets=None)
        if logger:
            if isinstance(proposals, list):
                proposal_shape = proposals[0].shape
            else:
                proposal_shape = proposals.shape
            logger.info(f"RegionProposal output shape: {type(proposals)} {proposal_shape}")
            logger.info(f"RegionProposal proposals: {rpn_losses if rpn_losses is not None else 'None'}")
            logger.info(f"RegionProposal forward OK!\n")
    except Exception as e:
        if logger:
            logger.error(f"RegionProposal error: {e}")
        all_ok = False

    try:
        head = Head(in_channels=256, num_classes=num_classes)
        head.eval()
        if isinstance(proposals, list):
            dummy_proposals = proposals
        else:
            dummy_proposals = [proposals]
        batch_dims = [(img_size, img_size)] * batch_size
        cls_probs, pred_bboxes, labels = head(features, dummy_proposals, batch_dims)
        if logger:
            logger.info(f"Head output (scores): {cls_probs.shape}")
            logger.info(f"Head output (bboxes): {pred_bboxes.shape}")
            logger.info(f"Head output (labels): {labels.shape}")
            logger.info(f"Head forward OK!\n")
    except Exception as e:
        if logger:
            logger.error(f"Head error: {e}")
        all_ok = False
    return all_ok


