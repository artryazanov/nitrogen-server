import sys
import os
import shutil
import json
import unittest
from unittest.mock import MagicMock, patch

# Robust Import Logic:
# 1. Check if 'serve' is already imported.
# 2. If not, we might need to mock missing dependencies (like zmq in local env) to let it import.
if 'serve' not in sys.modules:
    # Add scripts to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))
    
    # Mock dependencies if they are missing (to run locally without full env)
    # We check if we can import them, if not, we mock them in sys.modules
    required_modules = ['zmq', 'cv2', 'numpy', 'nitrogen', 'nitrogen.inference_session']
    for mod in required_modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

import serve

class TestDebugMode(unittest.TestCase):
    def setUp(self):
        self.debug_dir = "test_debug_output"
        if os.path.exists(self.debug_dir):
            shutil.rmtree(self.debug_dir)
        
        # Setup common mock behavior
        self.mock_session = MagicMock()
        self.mock_session.predict.return_value = {"action": [0, 1], "value": 0.5}
        self.mock_session.action_downsample_ratio = 1
        self.mock_session.info.return_value = "Mock Session"

    def tearDown(self):
        if os.path.exists(self.debug_dir):
            shutil.rmtree(self.debug_dir)

    def test_handle_request_debug_paths(self):
        # We use patch on 'serve.cv2' etc to safely mock the logic used inside serve.handle_request
        # This works even if serve loaded the real libraries.
        
        with patch('serve.cv2') as mock_cv2, \
             patch('serve.os.makedirs') as mock_makedirs, \
             patch('serve.np') as mock_np:
            
            # --- Setup Mocks --- as before
            mock_cv2.IMREAD_COLOR = 1
            mock_cv2.COLOR_BGR2RGB = 4
            mock_cv2.COLOR_RGB2BGR = 4
            
            # Side effect for imwrite to create a dummy file
            def side_effect_imwrite(filename, img):
                # Ensure directory is created (since we are patching os.makedirs below?) 
                # Wait, if we patch os.makedirs, serve.py won't create dir.
                # Let's NOT patch os.makedirs in the final call if we want files.
                # Actually, capturing the call args is enough for this test?
                # The user requirement was to save files. 
                # Checking file existence is better.
                # So we must NOT patch os.makedirs if we rely on serve.py creating it.
                # But we can patch os.makedirs to Just Do It and return.
                pass
            
            # If we want to check FILE EXISTENCE, we need real FS operations.
            # So we should NOT patch os.makedirs, os.path.join, etc.
            # We ONLY patch cv2.imwrite to write a dummy file instead of failing.
            
            pass # Close context to restart with clean list

        with patch('serve.cv2') as mock_cv2:
            # Setup dependencies
            mock_cv2.IMREAD_COLOR = 1
            mock_cv2.COLOR_BGR2RGB = 4
            mock_cv2.COLOR_RGB2BGR = 4
            mock_cv2.cvtColor.return_value = "dummy_bgr_image" # Valid enough for our fake imwrite

            def side_effect_imwrite(filename, img):
                 # Create the file on disk
                 dirname = os.path.dirname(filename)
                 if not os.path.exists(dirname):
                     os.makedirs(dirname, exist_ok=True)
                 with open(filename, 'wb') as f:
                     f.write(b'fake_img')
                 return True
            mock_cv2.imwrite.side_effect = side_effect_imwrite

            # Prepare inputs
            img = MagicMock()
            request = {"type": "predict", "image": "ignored", "some_param": 123}
            
            # Run
            serve.handle_request(
                self.mock_session, 
                request, 
                raw_image=img, 
                debug_mode=True, 
                debug_dir=self.debug_dir,
                original_image=img
            )

        # Assertions
        if not os.path.exists(self.debug_dir):
            self.fail(f"Debug directory {self.debug_dir} was not created")
            
        files = sorted(os.listdir(self.debug_dir))
        self.assertEqual(len(files), 4, f"Expected 4 files, found {files}")
        
        self.assertTrue(files[0].endswith("_1_received.png"))
        self.assertTrue(files[1].endswith("_2_params.json"))
        self.assertTrue(files[2].endswith("_3_processed.png"))
        self.assertTrue(files[3].endswith("_4_response.json"))
        
        # Check prefix
        prefix = files[0].split("_1_received.png")[0]
        for f in files:
            self.assertTrue(f.startswith(prefix))

if __name__ == '__main__':
    unittest.main()
