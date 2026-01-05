
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unittest.mock import MagicMock, Mock

# --- Aggressive Mocking of Missing Dependencies ---
# We mock these BEFORE any project imports happen.

class MockTensor(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = (1, 3, 256, 256) # Default dummy shape
        self.device = 'cpu'
        self.dtype = 'float32'
    
    def to(self, *args, **kwargs):
        return self
        
    def cpu(self):
        return self
        
    def numpy(self):
        # Return a dummy numpy array
        return MockArray()
        
    def squeeze(self, *args, **kwargs):
        return self
        
    def unsqueeze(self, *args, **kwargs):
        return self
    
    def __getitem__(self, key):
        return self

class MockArray(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = (256, 256, 3)
        self.dtype = 'uint8'
        
    def reshape(self, *args, **kwargs):
        return self
        
    def tolist(self):
        return [0]

# Mock torch
mock_torch = MagicMock()
mock_torch.Tensor = MockTensor
mock_torch.tensor = lambda data, **kwargs: MockTensor()
mock_torch.randn = lambda *args, **kwargs: MockTensor()
mock_torch.zeros = lambda *args, **kwargs: MockTensor()
mock_torch.ones = lambda *args, **kwargs: MockTensor()
mock_torch.cat = lambda *args, **kwargs: MockTensor()
mock_torch.load = lambda *args, **kwargs: {
    "ckpt_config": {
        "model_cfg": {"model_type": "nitrogen", "vision_encoder_name": "google/siglip-large-patch16-256"},
        "tokenizer_cfg": {"training": False}
    }, 
    "model": {}
}
mock_torch.bfloat16 = "bfloat16"
mock_torch.bool = "bool"
sys.modules['torch'] = mock_torch

# Mock numpy
mock_numpy = MagicMock()
mock_numpy.zeros = lambda *args, **kwargs: MockArray()
mock_numpy.frombuffer = lambda *args, **kwargs: MockArray()
mock_numpy.uint8 = "uint8"
sys.modules['numpy'] = mock_numpy

# Mock other ML/System libs
sys.modules['transformers'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['einops'] = MagicMock()
sys.modules['polars'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['zmq'] = MagicMock()
sys.modules['peft'] = MagicMock()

# --- Mocking nitrogen specific internals that might be hard to load ---

# Mock nitrogen.flow_matching_transformer.nitrogen
mock_fmt = MagicMock()
mock_fmt_nitrogen = MagicMock()
mock_fmt.nitrogen = mock_fmt_nitrogen
# We need NitroGen_Config to be a class we can instantiate or mock
class MockNitroGenConfig(MagicMock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
mock_fmt_nitrogen.NitroGen = MagicMock()
mock_fmt_nitrogen.NitroGen_Config = MockNitroGenConfig
sys.modules['nitrogen.flow_matching_transformer'] = mock_fmt
sys.modules['nitrogen.flow_matching_transformer.nitrogen'] = mock_fmt_nitrogen

# Mock nitrogen.mm_tokenizers
mock_mm = MagicMock()
class MockTokenizerConfig(MagicMock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
mock_mm.NitrogenTokenizerConfig = MockTokenizerConfig
mock_mm.NitrogenTokenizer = MagicMock()
mock_mm.Tokenizer = MagicMock()
sys.modules['nitrogen.mm_tokenizers'] = mock_mm

# Mock nitrogen.cfg
mock_cfg = MagicMock()
class MockCkptConfig(MagicMock): # Pydantic model mocks are tricky, using MagicMock
     def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if k == 'model_cfg' and isinstance(v, dict):
                v = MockNitroGenConfig(**v)
            if k == 'tokenizer_cfg' and isinstance(v, dict):
                v = MockTokenizerConfig(**v)
            setattr(self, k, v)
     
     @staticmethod
     def model_validate(obj):
         return MockCkptConfig(**obj) if isinstance(obj, dict) else obj
         
     def model_dump(self, *args, **kwargs):
         # Return a dict representation of the mock attributes
         return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}

class MockModalityConfig(MagicMock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_per_sample = kwargs.get('frame_per_sample', 16)
        self.action_interleaving = kwargs.get('action_interleaving', True)

mock_cfg.CkptConfig = MockCkptConfig
mock_cfg.ModalityConfig = MockModalityConfig
sys.modules['nitrogen.cfg'] = mock_cfg
sys.modules['nitrogen.shared'] = MagicMock()
sys.modules['nitrogen.shared'].PATH_REPO = "/tmp"

# --- Tests Imports (NOW we can safely import) ---
import pytest
from nitrogen.inference_session import InferenceSession

@pytest.fixture
def mock_ckpt_config():
    """Create a mock checkpoint configuration."""
    modality_cfg = MockModalityConfig(
        frame_per_sample=32,
        action_interleaving=True
    )
    
    tokenizer_cfg = MockTokenizerConfig(
        vocab_size=1000,
        training=False,
        game_mapping_cfg=None
    )
    
    model_cfg = MockNitroGenConfig(
        vision_encoder_name="openai/clip-vit-base-patch32",
        hidden_size=768
    )
    
    ckpt_config = MockCkptConfig(
        modality_cfg=modality_cfg,
        tokenizer_cfg=tokenizer_cfg,
        model_cfg=model_cfg
    )
    return ckpt_config

@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    # Mock return values for prediction methods
    model.get_action.return_value = MockTensor()
    model.get_action_with_cfg.return_value = MockTensor()
    return model

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    
    # Mock encode to return a dict of tensors
    tokenizer.encode.return_value = {
        "frames": MockTensor(),
        "buttons": MockTensor()
    }
    
    # Mock decode to return a dictionary of actions
    tokenizer.decode.return_value = {
        "buttons": MockTensor(),
        "j_left": MockTensor(),
        "j_right": MockTensor()
    }
    
    return tokenizer

@pytest.fixture
def mock_img_proc():
    """Create a mock image processor."""
    img_proc = MagicMock()
    # Mock processing result
    img_proc.return_value = {
        "pixel_values": MockTensor()
    }
    return img_proc

@pytest.fixture
def inference_session(mock_model, mock_tokenizer, mock_img_proc, mock_ckpt_config):
    """Create an InferenceSession instance with mocks."""
    session = InferenceSession(
        model=mock_model,
        ckpt_path="dummy_path.pt",
        tokenizer=mock_tokenizer,
        img_proc=mock_img_proc,
        ckpt_config=mock_ckpt_config,
        game_mapping={"game1": 1},
        selected_game="game1",
        old_layout=False,
        cfg_scale=1.5,
        action_downsample_ratio=1,
        context_length=16
    )
    return session
