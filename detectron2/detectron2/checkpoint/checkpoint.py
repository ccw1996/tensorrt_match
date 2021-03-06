import pickle
from fvcore.common.file_io import PathManager

from .detection_checkpoint import DetectionCheckpointer


class YOLOFCheckpointer(DetectionCheckpointer):
    """
    YOLOFCheckpoint to delete the key `weight_order` in pretrained models.
    """

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info(
                    "Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {
                    k: v
                    for k, v in data.items() if not k.endswith("_momentum")
                }
                # delete "weight_order"
                if "weight_order" in data:
                    del data["weight_order"]
                return {
                    "model": data,
                    "__author__": "Caffe2",
                    "matching_heuristics": True
                }

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded
