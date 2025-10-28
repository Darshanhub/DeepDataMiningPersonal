from typing import Callable, Dict, List, Optional, Union
import torch
from torch import nn, Tensor
import torchvision
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.models import resnet #, resnet50, ResNet50_Weights
from torchvision.models import get_model, get_model_weights, get_weight, list_models

# Import CBAM attention module
try:
    from DeepDataMiningLearning.detection.attention_modules import CBAM
    CBAM_AVAILABLE = True
except ImportError:
    print("[WARNING] CBAM module not available. Install attention_modules.py")
    CBAM_AVAILABLE = False

def get_backbone(model_name: str,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
    weights_enum = get_model_weights(model_name)
    weights = weights_enum.DEFAULT #IMAGENET1K_V1
    #weights = ResNet50_Weights.DEFAULT
    if model_name.startswith('resnet'):
        backbone = resnet.__dict__[model_name](weights=weights, norm_layer=norm_layer)
    elif model_name.startswith('swin'):
        backbone = get_model(model_name)
    else:
        backbone = get_model(model_name)

    return backbone
    # weights_backbone = ResNet50_Weights.verify(weights)
    # backbone = resnet50(weights=weights_backbone, progress=True)


class MyBackboneWithFPN(nn.Module):
    def __init__(
        self,
        model_name: str, #= 'resnet50'
        trainable_layers: int,
        #return_layers: Dict[str, str],
        #in_channels_list: List[int],
        out_channels: int = 256, #the number of channels in the FPN
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        weights_enum = get_model_weights(model_name) #ResNet152_Weights
        weights = weights_enum.DEFAULT #ResNet152_Weights.IMAGENET1K_V2
        #weights = ResNet50_Weights.DEFAULT
        backbone = resnet.__dict__[model_name](weights=weights, norm_layer=norm_layer)
        # weights_backbone = ResNet50_Weights.verify(weights)
        # backbone = resnet50(weights=weights_backbone, progress=True)

        #trainable_layers =2
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers] #trainable_layers=0=>layers_to_train=[]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        
        returned_layers = [1, 2, 3, 4]
        #return_layers (Dict[name, new_name]): a dict containing the names of the modules for which the activations will be returned as the key of the dict
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)} #{'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_stage2 = backbone.inplanes // 8 #2048//8=256
        #in_channels_list:List[int] number of channels for each feature map that is returned, in the order they are present in the OrderedDict
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        #[256, 512, 1024, 2048]
        # BackboneWithFPN(
        #     backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        # )
        #return_layers={'layer1': 'feat1', 'layer3': 'feat2'} #[name, new_name]
        #https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
        self.body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        # >>> out = new_m(torch.rand(1, 3, 224, 224))
        #     >>> print([(k, v.shape) for k, v in out.items()])
        #     >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        #     >>>      ('feat2', torch.Size([1, 256, 14, 14]))]

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x) #[16, 3, 800, 1344]
        x = self.fpn(x)
        return x
    
    #not used
    def create_fpnbackbone(self, backbone, trainable_layers):
        #backbone = get_model(backbone_modulename, weights="DEFAULT")
        trainable_layers =2
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        
        extra_blocks = LastLevelMaxPool()
        returned_layers = [1, 2, 3, 4]
        #return_layers (Dict[name, new_name]): a dict containing the names of the modules for which the activations will be returned as the key of the dict
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        in_channels_stage2 = backbone.inplanes // 8
        #in_channels_list:List[int] number of channels for each feature map that is returned, in the order they are present in the OrderedDict
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        #the number of channels in the FPN
        out_channels = 256
        # BackboneWithFPN(
        #     backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        # )
        #return_layers={'layer1': 'feat1', 'layer3': 'feat2'} #[name, new_name]
        #https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
        body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        # >>> out = new_m(torch.rand(1, 3, 224, 224))
        #     >>> print([(k, v.shape) for k, v in out.items()])
        #     >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        #     >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
        
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=None,
        )
        return body, fpn


class MyBackboneWithFPN_CBAM(nn.Module):
    """
    ResNet Backbone with Feature Pyramid Network (FPN) and CBAM Attention.
    
    This is an enhanced version of MyBackboneWithFPN that integrates CBAM
    (Convolutional Block Attention Module) into the backbone layers for
    improved feature representation in object detection.
    
    Architecture:
        ResNet (with CBAM) → FPN → Detection Head
    
    CBAM modules are added to:
        - layer3 (1024 channels) - for medium-scale features
        - layer4 (2048 channels) - for high-level semantic features
    
    Args:
        model_name (str): Name of the ResNet model ('resnet50', 'resnet152', etc.)
        trainable_layers (int): Number of layers to fine-tune (0 = freeze all)
        out_channels (int): Number of output channels in FPN (default: 256)
        extra_blocks (Optional[ExtraFPNBlock]): Additional FPN blocks (default: LastLevelMaxPool)
        norm_layer (Optional[Callable]): Normalization layer (default: None = BatchNorm2d)
        cbam_reduction (int): Channel reduction ratio for CBAM (default: 16)
        cbam_layers (List[str]): Which layers to add CBAM to (default: ['layer3', 'layer4'])
    
    Example:
        >>> backbone = MyBackboneWithFPN_CBAM('resnet152', trainable_layers=0, out_channels=256)
        >>> x = torch.randn(8, 3, 800, 800)
        >>> features = backbone(x)
        >>> print({k: v.shape for k, v in features.items()})
    """
    def __init__(
        self,
        model_name: str,
        trainable_layers: int,
        out_channels: int = 256,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        cbam_reduction: int = 16,
        cbam_layers: List[str] = None,
    ) -> None:
        super().__init__()
        
        if not CBAM_AVAILABLE:
            raise ImportError(
                "CBAM module not available. Please ensure attention_modules.py is in the correct path."
            )
        
        # Default CBAM layers
        if cbam_layers is None:
            cbam_layers = ['layer3', 'layer4']
        
        # Load pretrained ResNet backbone
        weights_enum = get_model_weights(model_name)
        weights = weights_enum.DEFAULT
        backbone = resnet.__dict__[model_name](weights=weights, norm_layer=norm_layer)
        
        # Freeze layers based on trainable_layers parameter
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        
        # Add CBAM modules to specified layers
        print(f"\n🔍 [CBAM] Adding attention modules to: {cbam_layers}")
        for layer_name in cbam_layers:
            if hasattr(backbone, layer_name):
                layer = getattr(backbone, layer_name)
                # Get the number of output channels for this layer
                if layer_name == 'layer1':
                    channels = 256 if 'resnet50' in model_name or 'resnet101' in model_name or 'resnet152' in model_name else 64
                elif layer_name == 'layer2':
                    channels = 512
                elif layer_name == 'layer3':
                    channels = 1024
                elif layer_name == 'layer4':
                    channels = 2048
                else:
                    print(f"[WARNING] Unknown layer: {layer_name}, skipping CBAM")
                    continue
                
                # Wrap the layer with CBAM
                enhanced_layer = self._add_cbam_to_layer(layer, channels, cbam_reduction)
                setattr(backbone, layer_name, enhanced_layer)
                print(f"✅ [CBAM] Added to {layer_name} ({channels} channels, reduction={cbam_reduction})")
        
        # Setup FPN
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        
        returned_layers = [1, 2, 3, 4]
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        
        # Create body with intermediate layer getter
        self.body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        # Create FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels
        
        # Store CBAM info for reporting
        self.cbam_layers = cbam_layers
        self.cbam_reduction = cbam_reduction
    
    def _add_cbam_to_layer(self, layer: nn.Module, channels: int, reduction: int) -> nn.Sequential:
        """
        Add CBAM attention module after each bottleneck block in a ResNet layer.
        
        Args:
            layer: ResNet layer (Sequential of Bottleneck blocks)
            channels: Number of channels in the layer output
            reduction: CBAM channel reduction ratio
        
        Returns:
            nn.Sequential: Enhanced layer with CBAM modules
        """
        enhanced_blocks = []
        
        for i, block in enumerate(layer):
            # Add the original bottleneck block
            enhanced_blocks.append(block)
            
            # Add CBAM attention after the block
            cbam = CBAM(channels, reduction=reduction)
            enhanced_blocks.append(cbam)
        
        return nn.Sequential(*enhanced_blocks)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass through backbone with CBAM attention and FPN.
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Dictionary of feature maps at different scales:
                '0': P2 (stride 4)
                '1': P3 (stride 8)
                '2': P4 (stride 16)
                '3': P5 (stride 32)
                'pool': P6 (stride 64, if using LastLevelMaxPool)
        """
        x = self.body(x)  # Extract features with CBAM attention
        x = self.fpn(x)    # Build feature pyramid
        return x
    
    def get_attention_info(self) -> Dict[str, any]:
        """
        Get information about CBAM configuration.
        
        Returns:
            Dictionary containing CBAM configuration details
        """
        return {
            'attention_type': 'CBAM',
            'layers_with_attention': self.cbam_layers,
            'reduction_ratio': self.cbam_reduction,
            'total_cbam_modules': sum(
                sum(1 for m in getattr(self.body, layer).modules() if isinstance(m, CBAM))
                for layer in self.cbam_layers if hasattr(self.body, layer)
            )
        }


import os
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.") #pip install -q torchinfo

def remove_classificationheader(model, num_removeblock):
    modulelist=model.children() #resnet50(pretrained=True).children()
    num_removeblock = 0-num_removeblock #-2
    newbackbone = nn.Sequential(*list(modulelist)[:num_removeblock])
    return newbackbone


if __name__ == "__main__":
    os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'
    DATAPATH='/data/cmpe249-fa23/torchvisiondata/'

    #model_name = 'resnet50' #["layer4", "layer3", "layer2", "layer1", "conv1"]
    #model_name = 'resnet152' #["layer4", "layer3", "layer2", "layer1", "conv1"]
    #https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py
    model_name = 'swin_s' # 'avgpool','flatten','head'
    backbone = get_model(model_name, weights="DEFAULT")
    backbone=remove_classificationheader(backbone, 3)
    summary(model=backbone, 
        input_size=(1, 3, 64, 64), #(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ) 

    trainable_layers = 2
    out_channels = 256
    model = MyBackboneWithFPN(model_name,trainable_layers, out_channels)
    x=torch.rand(1,3,64,64) #image.tensors #[2, 3, 800, 1312] list of tensors x= torch.rand(1,3,64,64)
    output = model(x) 
    print([(k, v.shape) for k, v in output.items()])
    #[('0', torch.Size([1, 256, 16, 16])), ('1', torch.Size([1, 256, 8, 8])), ('2', torch.Size([1, 256, 4, 4])), ('3', torch.Size([1, 256, 2, 2])), ('pool', torch.Size([1, 256, 1, 1]))]
