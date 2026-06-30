"""
4DAGE OBJ 提取服务器
运行本地 HTTP 服务器，接收浏览器 POST 的顶点/索引数据，写入 OBJ 文件
"""
import struct
import base64
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

OBJ_DIR = os.path.dirname(os.path.abspath(__file__)) + '/obj'
os.makedirs(OBJ_DIR, exist_ok=True)

class ExtractHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len)
        data = json.loads(body)
        
        model_id = data['model_id']
        pos_b64 = data.get('pos_b64', '')
        idx_b64 = data.get('idx_b64', '')
        meta = data.get('meta', {})
        
        vc = meta.get('vc', 0)
        ic = meta.get('ic', 0)
        its = meta.get('its', 2)
        stride = meta.get('stride', 32)
        pos_slot = meta.get('pos_slot', 0)
        
        print(f'[{model_id}] 接收: vc={vc}, ic={ic}, stride={stride}, its={its}, slot={pos_slot}')
        
        if not pos_b64:
            self.send_error(400, 'Missing pos_b64')
            return
        
        try:
            pos_bytes = base64.b64decode(pos_b64)
            total_floats = vc * (stride // 4)
            all_floats = struct.unpack(f'<{total_floats}f', pos_bytes[:total_floats * 4])
            positions = []
            for i in range(vc):
                x = all_floats[i * (stride // 4) + pos_slot * 3]
                y = all_floats[i * (stride // 4) + pos_slot * 3 + 1]
                z = all_floats[i * (stride // 4) + pos_slot * 3 + 2]
                positions.append((x, y, z))
            
            triangles = []
            if idx_b64:
                idx_bytes = base64.b64decode(idx_b64)
                fmt_char = 'H' if its == 2 else 'I'
                indices = struct.unpack(f'<{ic}{fmt_char}', idx_bytes[:ic * its])
                for t in range(0, ic - 2, 3):
                    triangles.append((indices[t] + 1, indices[t+1] + 1, indices[t+2] + 1))
            
            obj_path = f'{OBJ_DIR}/{model_id}.obj'
            with open(obj_path, 'w', encoding='utf-8') as f:
                f.write(f'# 4DAGE Model: {model_id}\n')
                f.write(f'# vertices: {vc}, indices: {ic}\n\n')
                for x, y, z in positions:
                    f.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
                for a, b, c in triangles:
                    f.write(f'f {a} {b} {c}\n')
            
            obj_size = os.path.getsize(obj_path)
            print(f'[{model_id}] OK: {obj_path} ({obj_size:,} bytes)')
            
            resp = {'status': 'ok', 'file': obj_path, 'size': obj_size}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        except Exception as e:
            print(f'[{model_id}] ERROR: {e}')
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))
    
    def do_GET(self):
        if self.path == '/ping':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass

if __name__ == '__main__':
    port = 8765
    server = HTTPServer(('127.0.0.1', port), ExtractHandler)
    print(f'Server: http://127.0.0.1:{port}')
    print(f'Output: {OBJ_DIR}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Stopped')
