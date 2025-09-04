__version__ = "1.0.0"

from .modeling_bert import (BertConfig, BertModel,
                       load_tf_weights_in_bert)

from .modeling_graphormer import Graphormer

from .e2e_body_network import Graphormer_Body_Network

from .e2e_hand_network import Graphormer_Hand_Network

CONFIG_NAME = "config.json"

from .modeling_utils import (
    WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    PretrainedConfig,
    PreTrainedModel,
    Conv1D,
)

# Ensure compatibility across different function names
try:
    from .modeling_utils import prune_layer  # type: ignore
except Exception:  # ImportError or AttributeError
    from .modeling_utils import prune_linear_layer as prune_layer  # type: ignore

from .file_utils import (PYTORCH_PRETRAINED_BERT_CACHE)
