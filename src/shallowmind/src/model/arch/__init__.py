
from .base import BaseArch
# from .base_encoder_decoder import BaseEncoderDecoder
# from .base_gan import BaseGAN
# from .base_vae import BaseVAE
# from .base_agent import BaseAgent
# from .not_use.EEGidentifierCL import EEGidentifierCLV0
from .EEGTransClassifer import EEGTransClassifer
# from .EEGTransClassiferAtt import EEGTransClassiferAtt
# from .SOFIN import SOFIN_arch
from .FuzzyTransformer import FuzzyTransformer_arch

# from .EEGTransClassifer_DecoderInputSubj2 import EEGTransClassifer_DecoderInputSubj2
__all__ = [
    "BaseArch",
    # "BaseEncoderDecoder", 
    # "BaseGAN", 
    # "BaseVAE", 
    # "BaseAgent", 
    # "EEGidentifierCLV0", 
    "EEGTransClassifer",
    # "EEGTransClassiferAtt",
    # "SOFIN_arch",
    "FuzzyTransformer_arch",
    'EEGTransClassifer_explainable',

    # "EEGTransClassifer_DecoderInputSubj2",

    ]