import socket               # Import socket module
import time
import cv2
from PIL import Image
import numpy

class Bridge:

    def __init__(self):
        self.time = 0
        self.x = 0
        self.y = 0

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
        if sent == 0:
            raise RuntimeError("socket connection broken")
    
    def askAndYouShalReceive(self, msg):
        self.sendAndForget(msg)
        chunk = self.sock.recv(1048)
        if chunk == '':
            raise RuntimeError("socket connection broken")
        return chunk
    
    def getScreen(self):
        chunk = self.askAndYouShalReceive("screen:single")
        if chunk == 'ready':
            screen = cv2.imread('screen.png')
        return screen

    def getPixelScreen(self):
        screen = self.getScreen()
        pixelSize = 8
        image = Image.fromarray(screen)
        image = image.resize((image.size[0]/pixelSize, image.size[1]/pixelSize), Image.NEAREST)
        # image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
        return numpy.array(image)

    def cord(self):
        xy = self.askAndYouShalReceive("location:get")
        x, y = xy.split(",")
        return int(x), int(y)

    def reset(self):
        self.sendAndForget("reset:true")

    def step(self, action):
        self.askAndYouShalReceive("key:" + action);
        x, y = self.cord()
        screen = None
        # screen = self.getScreen()
        reward = int(self.askAndYouShalReceive("points:get"))
        sdone = self.askAndYouShalReceive("done:get")
        done = False
        if sdone == 'True':
            done = True

        if self.x != x and self.y != y:
            self.time+=1
        self.x = x
        self.y = y
        return x, y, screen, reward, done
    
if __name__ == "__main__":
    bridge = Bridge()
    bridge.connectToSocket()
    screen = bridge.getScreen()
    print(screen.shape)
    img = Image.fromarray(screen, 'RGB')
    img.show()


    #bridge.sendAndForget("key:start");
    #time.sleep(1);
    #bridge.sendAndForget("skip:100");
    #while True:
    #    bridge.sendAndForget("key:right");
    #    bridge.sendAndForget("skip:31");
    #    time.sleep(1);
    #    bridge.sendAndForget("key:left");
    #    bridge.sendAndForget("skip:31");
    #    time.sleep(2);
