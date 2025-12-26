import sys
import os
import shutil
import json
import unittest
from unittest.mock import MagicMock, patch

# Mock modules before import
mock_cv2 = MagicMock()
mock_numpy = MagicMock()
mock_nitrogen = MagicMock()
mock_zmq = MagicMock()

sys.modules["cv2"] = mock_cv2
sys.modules["numpy"] = mock_numpy
sys.modules["nitrogen"] = mock_nitrogen
sys.modules["nitrogen.inference_session"] = mock_nitrogen
sys.modules["zmq"] = mock_zmq

# Setup mock constants that might be used
mock_cv2.IMREAD_COLOR = 1
mock_cv2.COLOR_BGR2RGB = 4
mock_cv2.COLOR_RGB2BGR = 4
mock_numpy.uint8 = "uint8"

# Add scripts to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

# Import serve
try:
    import serve
except ImportError as e:
    print(f"ImportError during serve import: {e}")
    sys.exit(1)

# Mock imwrite to actually create a file so we can check for existence
def side_effect_imwrite(filename, img):
    with open(filename, 'wb') as f:
        f.write(b'fake_image_data')
    return True

mock_cv2.imwrite.side_effect = side_effect_imwrite

class TestDebugMode(unittest.TestCase):
    def setUp(self):
        self.debug_dir = "test_debug_output"
        if os.path.exists(self.debug_dir):
            shutil.rmtree(self.debug_dir)
        
        # Mock session
        self.session = MagicMock()
        self.session.predict.return_value = {"action": [0, 1], "value": 0.5}
        self.session.action_downsample_ratio = 1
        self.session.info.return_value = "Mock Session"

    def tearDown(self):
        if os.path.exists(self.debug_dir):
            shutil.rmtree(self.debug_dir)

    def test_handle_request_debug(self):
        # Create a dummy image (mock object acting as numpy array)
        img = MagicMock()
        
        request = {"type": "predict", "image": "ignored_in_tcp_mode", "some_param": 123}
        
        # Call handle_request
        serve.handle_request(
            self.session, 
            request, 
            raw_image=img, 
            debug_mode=True, 
            debug_dir=self.debug_dir,
            original_image=img
        )
        
        # Check files
        if not os.path.exists(self.debug_dir):
            self.fail(f"Debug directory {self.debug_dir} was not created")
            
        files = sorted(os.listdir(self.debug_dir))
        # Expected: _1_received.png, _2_params.json, _3_processed.png, _4_response.json
        self.assertEqual(len(files), 4, f"Expected 4 files, found {files}")
        
        # Check naming convention
        self.assertTrue(files[0].endswith("_1_received.png"))
        self.assertTrue(files[1].endswith("_2_params.json"))
        self.assertTrue(files[2].endswith("_3_processed.png"))
        self.assertTrue(files[3].endswith("_4_response.json"))
        
        # Check prefix matching
        prefix = files[0].split("_1_received.png")[0]
        # Regex or simple check for timestamp format YYYYMMDD_HHMMSS_ffffff
        # Just checking they all share the prefix
        for f in files:
            self.assertTrue(f.startswith(prefix))
            
        # Verify JSON content
        with open(os.path.join(self.debug_dir, files[1]), 'r') as f:
            params = json.load(f)
            self.assertEqual(params["some_param"], 123)
            
        with open(os.path.join(self.debug_dir, files[3]), 'r') as f:
            resp = json.load(f)
            self.assertEqual(resp["action"], [0, 1])

if __name__ == '__main__':
    unittest.main()
