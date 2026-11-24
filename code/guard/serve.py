"""Typed inference API. Bind locally or put an authenticated TLS proxy in front."""
import argparse
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from jev_inference import JevClassifier


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='checkpoint')
    parser.add_argument('--base-model',default='Qwen/Qwen3.5-2B');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=29660)
    args=parser.parse_args();key=os.environ.get('JEV_API_KEY','')
    if args.host not in ('127.0.0.1','localhost','::1') and not key:
        parser.error('Non-loopback serving requires JEV_API_KEY and an HTTPS reverse proxy')
    model=JevClassifier(args.checkpoint,args.base_model,args.device);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            wire=json.dumps(value).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(wire)))
            self.end_headers();self.wfile.write(wire)
        def authorized(self):
            return not key or secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+key)
        def do_GET(self):
            if not self.authorized():return self.reply(401,{'error':'API key required'})
            if self.path!='/health':return self.reply(404,{'error':'Not found'})
            self.reply(200,{'status':'ready','model':'jev-v3.2-2b','context_limit':16384,'text_only':True,'truncation':False})
        def do_POST(self):
            if not self.authorized():return self.reply(401,{'error':'API key required'})
            if self.path!='/predict':return self.reply(404,{'error':'Not found'})
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=250000:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(size));start=time.monotonic()
                with lock:result=model.predict(data['input'],data['options'],data.get('mode','single'),data.get('safety_applicable',False))
                result['latency_ms']=round((time.monotonic()-start)*1000,2);self.reply(200,result)
            except (ValueError,KeyError,TypeError) as exc:self.reply(400,{'error':str(exc)})
            except Exception:self.reply(503,{'error':'Inference unavailable'})
    print(json.dumps({'model':'jev-v3.2-2b','address':args.host,'port':args.port}),flush=True)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()


if __name__=='__main__':main()
