import zmq
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

def handle_request(session, request, raw_image=None):
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
            result = session.predict(image)
            return {"status": "ok", "pred": result, "repeat": session.action_downsample_ratio}
        return {"status": "error", "message": "Unknown type"}


def read_image_from_conn(conn, expected_size=None):
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
                     if img.shape[0] != 256 or img.shape[1] != 256:
                          img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                     return img
             except Exception:
                 pass
                 
             # Fallback: Assume Raw RGB 256x256x3
             expected_raw = 256 * 256 * 3
             if len(raw_data) == expected_raw:
                  return np.frombuffer(raw_data, dtype=np.uint8).reshape(256, 256, 3).copy()
        return None

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
                
                if actual_width != 256 or actual_height != 256:
                    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                    
                return img
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
        return None

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
        return np.frombuffer(raw_data, dtype=np.uint8).reshape(256, 256, 3).copy()
        
    return None

def run_zmq_server(session, port):
    """Runs the ZeroMQ server (original protocol)."""
    context = zmq.Context()
    socket_zmq = context.socket(zmq.REP)
    socket_zmq.bind(f"tcp://*:{port}")
    print(f"ZMQ Server running on port {port}", flush=True)
    
    while True:
        try:
            msg = socket_zmq.recv()
            req = pickle.loads(msg)
            res = handle_request(session, req)
            socket_zmq.send(pickle.dumps(res))
        except Exception as e:
            print(f"ZMQ Error: {e}")

def run_tcp_server(session, port):
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
                if req.get("type") == "predict":
                    expected_len = req.get("len")
                    img = read_image_from_conn(conn, expected_size=expected_len)
                    if img is None:
                        print("Incomplete or invalid image data received")
                        break

                # 3. Process and send JSON response
                res = handle_request(session, req, raw_image=img)
                
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
    
    args = parser.parse_args()

    session = InferenceSession.from_ckpt(args.ckpt)

    # Start TCP server in a daemon thread
    tcp_thread = threading.Thread(target=run_tcp_server, args=(session, args.tcp_port), daemon=True)
    tcp_thread.start()
    time.sleep(0.5)

    # Run ZMQ server in the main thread
    try:
        run_zmq_server(session, args.zmq_port)
    except KeyboardInterrupt:
        print("\nShutting down server...")
