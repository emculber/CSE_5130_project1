frame = 0
while true do
gui.savescreenshotas("frame" .. frame)
FCEU.frameadvance();
frame = frame + 1
end
