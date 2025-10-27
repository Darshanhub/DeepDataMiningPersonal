import traceback
import torch
print('torch', torch.__version__)
try:
    from DeepDataMiningLearning.detection.modules.attention import CBAM
    from DeepDataMiningLearning.detection.modules.block import C2fCBAM, DeformConv
    print('Imported CBAM, C2fCBAM, DeformConv')
    # instantiate small modules
    x = torch.randn(1, 64, 32, 32)
    try:
        cb = CBAM(64)
        y = cb(x)
        print('CBAM OK ->', y.shape)
    except Exception as e:
        print('CBAM instantiation error:')
        traceback.print_exc()
    try:
        c2f = C2fCBAM(64, 64)
        y2 = c2f(x)
        print('C2fCBAM OK ->', y2.shape)
    except Exception as e:
        print('C2fCBAM instantiation error:')
        traceback.print_exc()
    try:
        # DeformConv may require torchvision.ops.DeformConv2d support
        d = DeformConv(64, 128, 3, 2)
        x2 = torch.randn(1, 64, 64, 64)
        y3 = d(x2)
        print('DeformConv OK ->', y3.shape)
    except Exception as e:
        print('DeformConv instantiation/error:')
        traceback.print_exc()
except Exception as e:
    print('Top-level import failed:')
    traceback.print_exc()
