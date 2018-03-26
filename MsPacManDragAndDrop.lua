-- load namespace
local socket = require("socket")
-- create a TCP socket and bind it to the local host, at any port
local server = assert(socket.bind("*", 35029))
-- find out which port the OS chose for us
local ip, port = server:getsockname()
-- print a message informing what's up
print("Please telnet to localhost on port " .. port)
print("After connecting, you have 10s to enter a line to be echoed")


function handleSocket()
  local client = server:accept()
  print("Connected with the client")
  client:settimeout(0)
  local line, err = client:receive('*l')
  print(line)
  print(err)
  if not err then client:send(line .. "\n") end
  return line
end

client:close()

while true do
  request = handleSocket()
  print(request)
  
--	inpt = input.get();
--  inputs = joypad.get(1);
--  inputs['right'] = true;
--
--  print(inputs)
--  print("writing right")
--  joypad.set(1, inputs);
--
--  if inpt['ymouse'] ~= inpt['ymouse'] then
--    print(inpt)
--  end
--
--	-- mouse cursor
--	gui.line(inpt['xmouse'] - 2, inpt['ymouse'], inpt['xmouse'] + 2, inpt['ymouse'], "blue");
--	gui.line(inpt['xmouse'], inpt['ymouse'] - 2, inpt['xmouse'], inpt['ymouse'] + 2, "blue");
--
--	mspacman = { x = memory.readbyte(0x0060), y = memory.readbyte(0x0062) };
--
--	-- Ms. Pac-Man cursor
--	gui.line(mspacman['x'] - 2, mspacman['y'], mspacman['x'] + 2, mspacman['y'], "blue");
--	gui.line(mspacman['x'], mspacman['y'] - 2, mspacman['x'], mspacman['y'] + 2, "blue");
--
--	-- write new position for Ms. Pac-Man
--	if inpt['leftclick'] then
--		memory.writebyte(0x0060, inpt['xmouse']);
--		memory.writebyte(0x0062, inpt['ymouse']);
--	end
--
FCEU.frameadvance();
end
