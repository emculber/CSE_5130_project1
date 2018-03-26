#!/usr/bin/python           # This is client.py file

import socket               # Import socket module

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 35029                # Reserve a port for your service.

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)

sock.connect((host, port))

MSGLEN=10;
msg = "Hello World"

totalsent = 0
while totalsent < MSGLEN:
    sent = sock.send('Hello 1\n')
    print sent;
    if sent == 0:
        raise RuntimeError("socket connection broken")

variable = raw_input('input something!: ')

def myreceive(self):
    chunks = []
    bytes_recd = 0
    while bytes_recd < MSGLEN:
        chunk = self.sock.recv(min(MSGLEN - bytes_recd, 2048))
        if chunk == '':
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd = bytes_recd + len(chunk)
    return ''.join(chunks)
