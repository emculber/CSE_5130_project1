local socket = require("socket")
local server = assert(socket.bind("*", 35029))
local ip, port = server:getsockname()
local client = nil;
local skip = 0
local lastInput = nil

print("Listending over " .. ip .. " on port " .. port)

local function connectSocket()
  client = server:accept()
  print("Connected with the client")
  --client:settimeout(0)
  defaultKeys = joypad.get(1)
end

local function handleSocket()
  local line, err = client:receive()
  print("Reciving line " .. line)
  return line
end

local function killSocketConnection()
  client:close()
end

local function handleInput(input)
  print("Setting input " .. input)
  inputs = joypad.get(1)
  inputs[input] = true
  if(lastInput ~= nil) then
    inputs[lastInput] = false
  end
  joypad.set(1, inputs)
  lastInput = value
end

local function getInputs()
  return "temp"
end

connectSocket()
while true do
  if skip == 0 then
    print("waiting for message")
    request = handleSocket()
    sentType, value = request:match("([^:]+):([^:]+)")

    if sentType == "key" then
      handleInput(value)
    elseif sentType == "get" then
      client:send(getInputs() .. "\n")
    elseif sentType == "skip" then
      print("Setting skip to " .. value)
      skip = tonumber(value)
    end
  else
    print("skipping socket listener")
    skip = skip - 1
  end
  
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
