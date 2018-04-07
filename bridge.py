import socket               # Import socket module
import time
import cv2
from PIL import Image

class Bridge:

    def __init__(self):
        self.data = []

    def connectToSocket(self):
        s = socket.socket()         # Create a socket object
        host = socket.gethostname() # Get local machine name
        port = 35029                # Reserve a port for your service.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        print("Atempting to connect with " + host + ":" + str(port))
        self.sock.connect((host, port))
        print("Connected with " + host + ":" + str(port))
    
    def sendAndForget(self, msg):
        sent = self.sock.send(msg + '\n')
        print(sent)
        if sent == 0:
            raise RuntimeError("socket connection broken")
    
    def askAndYouShalReceive(self, msg):
        self.sendAndForget(msg)
        chunk = self.sock.recv(1048)
        print(chunk)
        if chunk == '':
            raise RuntimeError("socket connection broken")
        return chunk
    
    def getScreen(self):
        chunk = self.askAndYouShalReceive("screen:single")
        if chunk == 'ready':
            print("Reading Screen")
            screen = cv2.imread('screen.png')
        return screen
    
if __name__ == "__main__":
    bridge = Bridge()
    bridge.connectToSocket()
    screen = bridge.getScreen()
    img = Image.fromarray(screen, 'RGB')
    img.show()


    #bridge.sendAndForget("key:start");
    #time.sleep(1);
    #bridge.sendAndForget("skip:100");
    #while True:
    #    bridge.sendAndForget("key:right");
    #    bridge.sendAndForget("skip:30");
    #    time.sleep(1);
    #    bridge.sendAndForget("key:left");
    #    bridge.sendAndForget("skip:30");
    #    time.sleep(1);
