import pytest
from unittest.mock import MagicMock, patch
from nitrogen.inference_session import load_model
import sys

# Since conftest.py mocks torch/nitrogen modules, we can assume they are available.
# We need to specifically test the branching logic in load_model.

@pytest.fixture
def mock_path():
    with patch("nitrogen.inference_session.Path") as mock_p:
        yield mock_p

def test_load_monolithic_checkpoint(mock_path):
    """Test loading a standard monolithic checkpoint."""
    # Setup mocks
    mock_path_obj = MagicMock()
    mock_path.return_value = mock_path_obj
    
    # Simulating it's NOT a LoRA adapter
    # CASE 1: Not a directory
    mock_path_obj.is_dir.return_value = False
    
    # We also need to mock _load_monolithic_checkpoint or the internal function calls
    # Since load_model calls _load_monolithic_checkpoint which calls torch.load etc.
    # And torch.load is mocked in conftest, it should return a dummy dict.
    
    # Call function
    model, tokenizer, img_proc, ckpt_config, game_mapping, downsample = load_model("dummy_ckpt.pt")
    
    # Assertions
    # Should have called torch.load on the checkpoint path
    # But wait, load_model logic: 
    # if path.is_dir() and (path / "adapter_config.json").exists(): ...
    
    assert model is not None, "Model should be loaded"
    # Verify we didn't use PeftModel
    assert not sys.modules['peft'].PeftModel.from_pretrained.called

def test_load_lora_checkpoint_success(mock_path):
    """Test loading a LoRA adapter with a base model."""
    # Setup mocks
    mock_path_obj = MagicMock()
    mock_path.return_value = mock_path_obj
    
    # Simulate it IS a LoRA adapter
    mock_path_obj.is_dir.return_value = True
    # (path / "adapter_config.json").exists() -> True
    mock_path_obj.__truediv__.return_value.exists.return_value = True
    
    base_model_path = "base_model.pt"
    ckpt_path = "lora_ckpt"
    
    # Mock peft
    mock_peft_model = sys.modules['peft'].PeftModel
    mock_peft_model.from_pretrained.return_value = MagicMock()
    
    # Call function
    load_model(ckpt_path, base_model_path=base_model_path)
    
    # Assertions
    # 1. Check if PeftModel.from_pretrained was called
    assert mock_peft_model.from_pretrained.called
    args, _ = mock_peft_model.from_pretrained.call_args
    # First arg should be the base model (which came from _load_monolithic_checkpoint)
    # Second arg should be ckpt_path
    assert args[1] == ckpt_path
    
    # 2. Check merge_and_unload called
    assert mock_peft_model.from_pretrained.return_value.merge_and_unload.called

def test_load_lora_checkpoint_missing_base(mock_path):
    """Test loading a LoRA adapter without a base model raises ValueError."""
    # Setup mocks
    mock_path_obj = MagicMock()
    mock_path.return_value = mock_path_obj
    
    # Simulate it IS a LoRA adapter
    mock_path_obj.is_dir.return_value = True
    mock_path_obj.__truediv__.return_value.exists.return_value = True
    
    # Call and expect error
    with pytest.raises(ValueError, match="no --base-model provided"):
        load_model("lora_ckpt", base_model_path=None)
