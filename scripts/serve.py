import zmq
import datetime
import os
import time
import argparse
import pickle
import struct
import json
import socket
import numpy as np
import cv2
import threading
from nitrogen.inference_session import InferenceSession

# Lock to ensure thread safety for the stateful InferenceSession
session_lock = threading.Lock()

def preprocess_image(img, mode="pad"):
    """
    Resizes image to 256x256 based on the mode:
    - stretch: simple resize (default old behavior)
    - crop: center crop to square, then resize
    - pad: pad with black to square, then resize (default new behavior)
    """
    target_size = (256, 256)
    # Check if image is valid
    if img is None:
        return None
        
    h, w = img.shape[:2]

    if mode == "stretch":
        if (w, h) != target_size:
            return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img

    elif mode == "crop":
        min_dim = min(h, w)
        if h != w:
            center_h, center_w = h // 2, w // 2
            half_dim = min_dim // 2
            start_h = max(0, center_h - half_dim)
            start_w = max(0, center_w - half_dim)
            end_h = start_h + min_dim
            end_w = start_w + min_dim
            img = img[start_h:end_h, start_w:end_w]
        
        if img.shape[:2] != (256, 256):
             return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img

    elif mode == "pad":
        max_dim = max(h, w)
        if h != w:
            top = (max_dim - h) // 2
            bottom = max_dim - h - top
            left = (max_dim - w) // 2
            right = max_dim - w - left
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        if img.shape[:2] != (256, 256):
             return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img
    
    # Fallback to pad if unknown mode
    if h != w:
        return preprocess_image(img, "pad")
    if img.shape[:2] != target_size:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

def handle_request(session, request, raw_image=None, debug_mode=False, debug_dir="debug", original_image=None):
    """Universal request handler for ZeroMQ+Pickle and TCP+JSON+RawBytes protocols."""
    with session_lock:
        if request["type"] == "reset":
            session.reset()
            return {"status": "ok"}
        elif request["type"] == "info":
            return {"status": "ok", "info": session.info()}
        elif request["type"] == "predict":
            # If this is a Pickle request, the image is already inside the object
            image = raw_image if raw_image is not None else request.get("image")
            
            # Save debug artifacts if enabled
            if debug_mode:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    prefix = f"{timestamp}"
                    
                    # 1. Received Image (Original)
                    if original_image is not None:
                        # Convert RGB back to BGR for cv2.imwrite
                        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_1_received.png"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
                    elif image is not None:
                         # Fallback if original not provided separately
                         cv2.imwrite(os.path.join(debug_dir, f"{prefix}_1_received.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                    # 2. JSON Parameters
                    with open(os.path.join(debug_dir, f"{prefix}_2_params.json"), "w") as f:
                        # Filter out large data if strictly needed, but request usually just has metadata + array
                        # We should be careful not to dump huge arrays in text. 
                        # The 'request' dict might contain the image if it's ZMQ pickle.
                        # Create a safe copy for logging
                        log_req = {k: v for k, v in request.items() if k != "image"}
                        json.dump(log_req, f, indent=4)
                    
                    # 3. Image sent to model (Processed)
                    if image is not None:
                         cv2.imwrite(os.path.join(debug_dir, f"{prefix}_3_processed.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                except Exception as e:
                    print(f"Debug logging error: {e}")

            result = session.predict(image)

            if debug_mode:
                try:
                     # 4. Model Response
                    with open(os.path.join(debug_dir, f"{prefix}_4_response.json"), "w") as f:
                        # Convert numpy types to native types for JSON serialization
                        def default_converter(o):
                            if isinstance(o, np.integer): return int(o)
                            if isinstance(o, np.floating): return float(o)
                            if isinstance(o, np.ndarray): return o.tolist()
                            raise TypeError
                        json.dump(result, f, indent=4, default=default_converter)
                except Exception as e:
                    print(f"Debug logging response error: {e}")

            return {"status": "ok", "pred": result, "repeat": session.action_downsample_ratio}
        return {"status": "error", "message": "Unknown type"}


def read_image_from_conn(conn, expected_size=None, resize_mode='pad'):
    """
    Reads an image from the connection.
    If expected_size is provided, reads exactly that many bytes.
    Otherwise, detects BMP format by checking for 'BM' signature.
    """
    if expected_size is not None:
        raw_data = b""
        while len(raw_data) < expected_size:
            target = expected_size - len(raw_data)
            chunk = conn.recv(target)
            if not chunk: 
                break
            raw_data += chunk
            
        if len(raw_data) == expected_size:
             # Try to decode as generic image (BMP, PNG, etc.) from memory
             img = np.frombuffer(raw_data, dtype=np.uint8)
             try:
                 img = cv2.imdecode(img, cv2.IMREAD_COLOR) # Using opencv to decode buffer is safer/easier
                 if img is not None:
                     # OpenCV loads as BGR. 
                     # We need RGB.
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                     original_img = img.copy()
                     img = preprocess_image(img, resize_mode)
                     return img, original_img
             except Exception:
                 pass
                 
             # Fallback: Assume Raw RGB 256x256x3
             expected_raw = 256 * 256 * 3
             if len(raw_data) == expected_raw:
                  img = np.frombuffer(raw_data, dtype=np.uint8).reshape(256, 256, 3).copy()
                  return img, img.copy()
        return None, None

    # 1. Peek/Read first 2 bytes to check for BMP signature 'BM'
    sig = b""
    while len(sig) < 2:
        chunk = conn.recv(2 - len(sig))
        if not chunk: return None
        sig += chunk
    
    is_bmp = (sig == b'BM')
    
    # === BMP PATH ===
    if is_bmp:
        # Read next 52 bytes to complete 54-byte header
        rest_header = b""
        while len(rest_header) < 52:
            chunk = conn.recv(52 - len(rest_header))
            if not chunk: return None # Broken stream
            rest_header += chunk
        
        header_data = sig + rest_header
        
        # Parse BMP header
        try:
            # File size is at offset 2 (4 bytes, little endian)
            file_size = struct.unpack_from('<I', header_data, 2)[0]
            
            width, height = struct.unpack_from('<ii', header_data, 18)
            
            is_bottom_up = height > 0
            actual_width = width
            actual_height = abs(height)
            
            # BMP aligns rows to 4-byte boundaries
            row_size = (actual_width * 3 + 3) & ~3
            pixel_data_size = row_size * actual_height
            
            # Read pixels + any extra metadata/padding
            # We already read 54 bytes (14 header + 40 info)
            # WAIT: We read 'sig' (2) + 'rest_header' (52) = 54 bytes.
            # So remaining is file_size - 54.
            remaining_bytes = file_size - 54
            
            raw_data = b""
            while len(raw_data) < remaining_bytes:
                chunk = conn.recv(remaining_bytes - len(raw_data))
                if not chunk: break
                raw_data += chunk
            
            if len(raw_data) == remaining_bytes:
                # We interpret the first 'pixel_data_size' bytes of raw_data as pixels
                # (Assuming pixel offset is 54, which is standard for 24-bit BMP)
                # Ideally we should read bfOffBits at offset 10 to be sure where pixels start.
                bfOffBits = struct.unpack_from('<I', header_data, 10)[0]
                
                # The pixel data starts at bfOffBits - 54 (since we consumed 54)
                pixel_start = bfOffBits - 54
                
                if pixel_start < 0 or pixel_start + pixel_data_size > len(raw_data):
                     # Fallback or error?
                     # If standard V3 header, offset is usually 54. 
                     # If we have color table, offset > 54.
                     pass 
                
                pixel_bytes = raw_data[pixel_start : pixel_start + pixel_data_size]
                
                img = np.frombuffer(pixel_bytes, dtype=np.uint8)
                
                # If there is alignment padding, remove it
                if row_size != actual_width * 3:
                     img = img.reshape(actual_height, row_size)[:, :actual_width * 3]
                
                img = img.reshape(actual_height, actual_width, 3)
                
                if is_bottom_up:
                    img = cv2.flip(img, 0)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                original_img = img.copy()
                
                if actual_width != 256 or actual_height != 256:
                    img = preprocess_image(img, resize_mode)
                    
                return img, original_img
        except struct.error:
            # Fallback to Raw if parsing fails? Unlikely if we got bytes.
            pass

    # === RAW PATH (Fallback or Non-BMP) ===
    # If we are here, it wasn't a BMP or we decided to treat as raw.
    # Note: If it WAS 'BM' but processing failed, we might have already consumed bytes.
    # But for now, let's assume if it starts with BM it IS a BMP. 
    # If signature was NOT BM, we have 2 bytes in 'sig'.
    
    if is_bmp:
        # If we failed BMP parsing but determined it was BMP, return None. 
        # (Mixed logic: 'BM' is strong indicator. If header is partial, connection is bad).
        return None, None

    # Raw expected size: 256x256x3 = 196608
    expected_bytes = 256 * 256 * 3
    
    # We already have 'sig' (2 bytes)
    raw_data = sig
    while len(raw_data) < expected_bytes:
        chunk = conn.recv(expected_bytes - len(raw_data))
        if not chunk: break
        raw_data += chunk
        
    if len(raw_data) == expected_bytes:
        # Assume Raw is already RGB and correctly oriented (256x256)
        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(256, 256, 3).copy()
        return img, img.copy()
        
    return None, None

def run_zmq_server(session, port, debug_mode=False, debug_dir="debug"):
    """Runs the ZeroMQ server (original protocol)."""
    context = zmq.Context()
    socket_zmq = context.socket(zmq.REP)
    socket_zmq.bind(f"tcp://*:{port}")
    print(f"ZMQ Server running on port {port}", flush=True)
    
    while True:
        try:
            msg = socket_zmq.recv()
            req = pickle.loads(msg)
            # For ZMQ, we don't distinguish original vs processed in quite the same way yet
            # as it's often sent pre-processed or we treat it as is.
            res = handle_request(session, req, debug_mode=debug_mode, debug_dir=debug_dir) 
            # Note: We didn't pipe flags to run_zmq_server yet or update its signature, 
            # but user request emphasizes "received image" which implies the TCP/file path mostly.
            # However, ZMQ is also a "request".
            # Currently handle_request defaults debug=False, so ZMQ won't debug unless we update this.
            # But the main usage seems to be the TCP text/json protocol for now.
            # Let's leave ZMQ without debug args for now unless required, or use global args if we pass them.
            # BETTER: Update run_zmq_server signature to take debug args.
            
            socket_zmq.send(pickle.dumps(res))
        except Exception as e:
            print(f"ZMQ Error: {e}")

def run_tcp_server(session, port, debug_mode=False, debug_dir="debug"):
    """Runs the simple TCP server (for BizHawk/Lua)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(('0.0.0.0', port))
    except OSError as e:
        print(f"Error binding TCP port {port}: {e}")
        return

    server.listen(1)
    print(f"Simple TCP Server (JSON+Bytes) running on port {port}", flush=True)
    
    while True:
        conn, addr = server.accept()
        # print(f"TCP Client connected from {addr}")
        try:
            while True:
                # 1. Read request header (JSON string until \n)
                # Read byte by byte until newline to avoid over-reading the pixel data
                line_bytes = b""
                while True:
                    char = conn.recv(1)
                    if not char: 
                        break # Connection closed or empty
                    if char == b'\n':
                        break
                    line_bytes += char
                
                if not line_bytes: 
                    break

                try:
                    req = json.loads(line_bytes.decode('utf-8'))
                except json.JSONDecodeError:
                    print("Invalid JSON received")
                    break

                img = None
                original_img = None
                
                if req.get("type") == "predict":
                    expected_len = req.get("len")
                    resize_mode = req.get("resize_mode", "pad")
                    img, original_img = read_image_from_conn(conn, expected_size=expected_len, resize_mode=resize_mode)
                    if img is None:
                        print("Incomplete or invalid image data received")
                        break

                # 3. Process and send JSON response
                res = handle_request(session, req, raw_image=img, debug_mode=debug_mode, debug_dir=debug_dir, original_image=original_img)
                
                # Convert numpy to lists for JSON
                if "pred" in res:
                    res["pred"] = {k: v.tolist() for k, v in res["pred"].items()}
                
                response_json = json.dumps(res)
                conn.sendall((response_json + "\n").encode('utf-8'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"TCP Connection error: {e}", flush=True)
        finally:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--zmq-port", type=int, default=5555, help="Port for ZeroMQ server")
    parser.add_argument("--tcp-port", type=int, default=5556, help="Port for Simple TCP server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save images and jsons")
    parser.add_argument("--debug-dir", type=str, default="debug", help="Directory to save debug files")
    
    args = parser.parse_args()

    session = InferenceSession.from_ckpt(args.ckpt)

    # Start TCP server in a daemon thread
    tcp_thread = threading.Thread(target=run_tcp_server, args=(session, args.tcp_port, args.debug, args.debug_dir), daemon=True)
    tcp_thread.start()
    time.sleep(0.5)

    # Run ZMQ server in the main thread
    try:
        run_zmq_server(session, args.zmq_port, debug_mode=args.debug, debug_dir=args.debug_dir)
    except KeyboardInterrupt:
        print("\nShutting down server...")
