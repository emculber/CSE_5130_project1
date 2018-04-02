local socket = require("socket")
local server = assert(socket.bind("*", 35029))
local ip, port = server:getsockname()
local client = nil;
local skip = 0
local lastInput = nil
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
  print("Reciving line " .. line)
  return line
end

local function killSocketConnection()
  print("Closing Connections")
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

connectSocket()
while true do
  if skip == 0 then
    print("waiting for message")
    request = handleSocket()
    sentType, value = request:match("([^:]+):([^:]+)")

    if sentType == "key" then
      handleInput(value)
    elseif sentType == "get" then
      client:send(getInputs())
    elseif sentType == "screen" then
      screenshot()
      client:send("ready")
      dontRender = true
    elseif sentType == "skip" then
      print("Setting skip to " .. value)
      skip = tonumber(value)
    elseif sentType == "close" then
      killSocketConnection()
    end
  else
    print("skipping socket listener")
    skip = skip - 1
  end
  if dontRender == false then
    FCEU.frameadvance();
  end
  dontRender = false
end
