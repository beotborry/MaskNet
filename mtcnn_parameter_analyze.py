from mtcnn.mtcnn import MTCNN
from utils import export_model_summary

network = MTCNN()

export_model_summary(network._pnet, "pnet")
export_model_summary(network._onet, "onet")
export_model_summary(network._rnet, "rnet")

network._pnet.summary()
network._onet.summary()
network._rnet.summary()