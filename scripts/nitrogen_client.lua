-- NitroGen Client for BizHawk (Native .NET Version)
-- FIX: Renamed 'client' variable to 'tcp' to avoid conflict with Emulator API

local luanet = _G.luanet
luanet.load_assembly("System")

-- Imports
local TcpClient = luanet.import_type("System.Net.Sockets.TcpClient")
local File = luanet.import_type("System.IO.File") 
local Encoding = luanet.import_type("System.Text.Encoding")

-- === CONFIGURATION ===
local HOST = "127.0.0.1"
local PORT = 5556
local TEMP_IMG_FILE = "nitrogen_temp.png"
local CONSOLE_TYPE = "NES" -- "SNES" or "NES"

-- === CONTROL MAPPING ===
-- Updated function: accepts ready tables of values for the current frame
local function apply_controls_frame(btn_slice, stick_slice)
    local joy = {}
    
    -- Parse stick (if data exists)
    local lx = stick_slice[1] or 0
    local ly = stick_slice[2] or 0
    
    -- Stick threshold (can be reduced to 0.3 if reaction is poor)
    local threshold = 0.5
    local stick_left  = lx < -threshold
    local stick_right = lx > threshold
    local stick_up    = ly < -threshold
    local stick_down  = ly > threshold

    -- IMPORTANT: btn_slice contains 21 values for the current frame
    if #btn_slice < 21 then return end

    if CONSOLE_TYPE == "SNES" then
        joy["P1 B"]      = btn_slice[6]  > 0.5 
        joy["P1 A"]      = btn_slice[19] > 0.5 
        joy["P1 Y"]      = btn_slice[21] > 0.5 
        joy["P1 X"]      = btn_slice[11] > 0.5 
        joy["P1 Up"]     = (btn_slice[5]  > 0.5) or stick_up
        joy["P1 Down"]   = (btn_slice[2]  > 0.5) or stick_down
        joy["P1 Left"]   = (btn_slice[3]  > 0.5) or stick_left
        joy["P1 Right"]  = (btn_slice[4]  > 0.5) or stick_right
        joy["P1 Start"]  = btn_slice[20] > 0.5 
        joy["P1 Select"] = btn_slice[1]  > 0.5 
        joy["P1 L"]      = btn_slice[8]  > 0.5 
        joy["P1 R"]      = btn_slice[15] > 0.5 
    elseif CONSOLE_TYPE == "NES" then
        joy["P1 A"]      = btn_slice[19] > 0.5 
        joy["P1 B"]      = btn_slice[6]  > 0.5 
        joy["P1 Up"]     = (btn_slice[5]  > 0.5) or stick_up
        joy["P1 Down"]   = (btn_slice[2]  > 0.5) or stick_down
        joy["P1 Left"]   = (btn_slice[3]  > 0.5) or stick_left
        joy["P1 Right"]  = (btn_slice[4]  > 0.5) or stick_right
        joy["P1 Start"]  = btn_slice[20] > 0.5 
        joy["P1 Select"] = btn_slice[1]  > 0.5 
    end
    
    joypad.set(joy)
end

-- Helper function to extract all numbers from a JSON block
local function extract_numbers(json_str, key)
    local values = {}
    -- Find start of block by key "key": [
    local s = string.find(json_str, "\"" .. key .. "\":%s*%[")
    if not s then return values end
    
    -- Rough but working method: read numbers starting from found position
    -- until we meet other keys or end
    -- Better to find the closing bracket of the array "]]"
    local sub = string.sub(json_str, s)
    local e = string.find(sub, "]]") -- End of outer array
    if e then
        sub = string.sub(sub, 1, e + 1)
        for v in string.gmatch(sub, "[%-%d%.]+") do
            table.insert(values, tonumber(v))
        end
    end
    return values
end

-- === MAIN LOGIC ===
console.clear()
console.log("Connecting to " .. HOST .. ":" .. PORT .. "...")

-- FIX: Rename variable to 'tcp' so we don't hide global 'client'
local tcp = TcpClient()
local success, err = pcall(function() 
    tcp:Connect(HOST, PORT) 
end)

if not success then
    console.log("Connection Failed: " .. tostring(err))
    return
end

-- Set timeout to 5s so the emulator doesn't hang if the server is "thinking"
tcp.ReceiveTimeout = 5000
tcp.SendTimeout = 5000

console.log("Connected!")
local stream = tcp:GetStream()
local resp_buffer = luanet.import_type("System.Byte[]")(4096)

while tcp.Connected do
    -- 1. Screenshot
    client.screenshot(TEMP_IMG_FILE)
    
    -- 2. Read Bytes & Send Header
    local file_bytes = File.ReadAllBytes(TEMP_IMG_FILE)
    local len = file_bytes.Length
    local json_header = string.format('{"type": "predict", "len": %d}\n', len)
    local header_bytes = Encoding.ASCII:GetBytes(json_header)
    
    stream:Write(header_bytes, 0, header_bytes.Length)
    stream:Write(file_bytes, 0, file_bytes.Length)
    
    -- 3. Receive & Process Loop
    local bytes_read = stream:Read(resp_buffer, 0, resp_buffer.Length)
    if bytes_read > 0 then
        local resp_str = Encoding.ASCII:GetString(resp_buffer, 0, bytes_read)
        
        -- Extract ALL numbers for buttons and stick
        local all_buttons = extract_numbers(resp_str, "buttons")
        local all_sticks  = extract_numbers(resp_str, "j_left")
        
        -- Calculate number of steps (frames) predicted by the model
        -- Each button step takes 21 numbers
        local num_steps = math.floor(#all_buttons / 21)
        
        if num_steps > 0 then
            gui.drawText(0, 0, "AI Steps: " .. num_steps, "green")
            
            -- Play the ENTIRE predicted sequence
            for i = 0, num_steps - 1 do
                -- Button slice for current step
                local btn_slice = {}
                for k = 1, 21 do
                    table.insert(btn_slice, all_buttons[i * 21 + k])
                end
                
                -- Stick slice for current step
                local stick_slice = {0, 0}
                if #all_sticks >= (i + 1) * 2 then
                    stick_slice[1] = all_sticks[i * 2 + 1]
                    stick_slice[2] = all_sticks[i * 2 + 2]
                end
                
                -- Apply controls and advance frame
                apply_controls_frame(btn_slice, stick_slice)
                emu.frameadvance()
            end
        else
            -- If response is empty or failed to parse, just skip frame
            emu.frameadvance()
        end
    end
end

tcp:Close()
console.log("Disconnected.")