local socket = require("socket")
local server = assert(socket.bind("*", 35029))
local ip, port = server:getsockname()
local client = nil;
local skip = 0
local lastInput = 'none'
local dontRender = false

print("Listending over " .. ip .. " on port " .. port)

local function connectSocket()
  client = server:accept()
  print("Connected with the client")
  --client:settimeout(0)
  defaultKeys = joypad.get(1)
end

local function handleSocket()
  local line, err = client:receive()
  --print("Reciving line " .. line)
  return line
end

local function killSocketConnection()
  print("Closing Connections")
  client:close()
end

local function handleInput(input)
  --print("Setting input " .. input)
  emu.message(input);
  inputs = joypad.get(1)
  if(lastInput ~= nil) then
    inputs[lastInput] = false
    joypad.set(1, inputs)
  end
  inputs[input] = true
  joypad.set(1, inputs)
  lastInput = value
end

local function getInputs()
  --return table_to_string(joypad.get(1))
  return "[up,left,right,down]"
end

local function screenshot()
  gui.savescreenshotas("screen.png")
end

function table_to_string(tbl)
  local result = "{"
  for k, v in pairs(tbl) do
    -- Check the key type (ignore any numerical keys - assume its an array)
    if type(k) == "string" then
      result = result.."[\""..k.."\"]".."="
    end

    -- Check the value type
    if type(v) == "table" then
      result = result..table_to_string(v)
    elseif type(v) == "boolean" then
      result = result..tostring(v)
    else
      result = result.."\""..v.."\""
    end
    result = result..","
  end
  -- Remove leading commas from the result
  if result ~= "" then
    result = result:sub(1, result:len()-1)
  end
  return result.."}"
end

function loadState()
  emu.softreset()
  state = savestate.create(1)
  savestate.load(state)
  emu.speedmode("normal")
end

function getPoints()
  d1 = memory.readbyte(0x0040)
  d2 = memory.readbyte(0x0041)
  d3 = memory.readbyte(0x0042)
  d4 = memory.readbyte(0x0043)
  d5 = memory.readbyte(0x0044)
  d6 = memory.readbyte(0x0045)

  points = d1 .. d2 .. d3 .. d4 .. d5 .. d6
  points = tonumber(points)
  return points
end

function getPellets()
  print("Reading Pellets")
  p = memory.readbyte(0x003d)
  print("Pellets: " .. p)
  return p
end

connectSocket()
while true do
  if skip == 0 then
    --print("waiting for message")
    request = handleSocket()
    sentType, value = request:match("([^:]+):([^:]+)")

    if sentType == "key" then
      handleInput(value)
      skip = 2
    elseif sentType == "get" then
      client:send(getInputs())
    elseif sentType == "screen" then
      screenshot()
      client:send("ready")
      dontRender = true
    elseif sentType == "location" then
      mspacman = { x = memory.readbyte(0x0060), y = memory.readbyte(0x0062) };
      client:send(mspacman['x'] .. "," .. mspacman['y'])
      dontRender = true
    elseif sentType == "points" then
      client:send(getPoints())
      dontRender = true
    elseif sentType == "pellets" then
      client:send(getPellets())
      dontRender = true
    elseif sentType == "reset" then
      loadState()
      dontRender = true
    elseif sentType == "done" then
      done = memory.readbyte(0x003f)
      if done == 2 then
        client:send('True')
      else
        client:send('False')
      end
      dontRender = true
    elseif sentType == "skip" then
      --print("Setting skip to " .. value)
      skip = tonumber(value)
    elseif sentType == "close" then
      killSocketConnection()
      connectSocket()
    else
      print("Unknown Key")
    end
  else
    --print("skipping socket listener")
    skip = skip - 1
    if skip == 0 then
      client:send("Finished")
    end
  end
  if dontRender == false then
    --emu.message(lastInput);
    FCEU.frameadvance();
  end
  dontRender = false
end
