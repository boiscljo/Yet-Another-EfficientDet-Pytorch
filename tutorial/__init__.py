from ..detectors.Detector import *
from .EffDet.model import getEfficientDetImpl
import torch
class EfficientDetector(Detector):
    module: torch.nn.Sequential
    def __init__(self) -> None:
        super(EfficientDetector,self).__init__(3,False)
        self.module = getEfficientDetImpl()
        pass
    def parameters(self, recurse: bool = True):
        return self.module.parameters()
    def state_dict(self):
        return self.state_dict()
    def load_state_dict(self, state_dict, strict: bool = True):
        return self.module.load_state_dict(self,state_dict,strict)

    def _forward(self, gray:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor, target=None, dataset=None):
        return Detection()
    
    
Detector.register("EfficientDetector",EfficientDetector)
