RUN SCRIPT
=========

1. Open FCEUX
2. Go to file and open ROM and select the Ms. Pac-Man (U) [!].nes
3. Go to file and Load Lua script and select bridge.lua
4. run python neatImp.py to run the NEAT algorithm

bridge.py methods
================
1. connectToSocket : handles the connection to the lua and emulator
2. sendAndForget : Sends message though he socket but does not recive a message back
3. askAndYouShalReceive : Sends a message though the socket and waits for a response
4. getScreen : returns a numpy array of the screen

Command to send
==============
1. key:start, key:right, key:left, etc. are all key presses. 
2. skip:1000 skips 1000 frames. the number can be anything
3. get:inputs returns the set of possible inputs
4. screen:single takes a screenshot and saves it as screen.png (Use getScreen method because you will have to load the image yourself if you use this)
