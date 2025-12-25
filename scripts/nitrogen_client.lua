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
local function apply_controls(buttons, lx, ly)
    local joy = {}
    
    -- Determine stick inputs (threshold 0.2 for better sensitivity)
    -- Y < -0.2 usually means UP, Y > 0.2 means DOWN
    local stick_threshold = 0.2
    local stick_left  = lx < -stick_threshold
    local stick_right = lx > stick_threshold
    local stick_up    = ly < -stick_threshold
    local stick_down  = ly > stick_threshold

    -- Button threshold 0.3 to catch "weak" presses
    local btn_thresh = 0.3

    if CONSOLE_TYPE == "SNES" then
        joy["P1 B"]      = buttons[6]  > btn_thresh 
        joy["P1 A"]      = buttons[19] > btn_thresh 
        joy["P1 Y"]      = buttons[21] > btn_thresh 
        joy["P1 X"]      = buttons[11] > btn_thresh 
        
        -- OR conditions: D-Pad button OR stick direction
        joy["P1 Up"]     = (buttons[5]  > btn_thresh) or stick_up
        joy["P1 Down"]   = (buttons[2]  > btn_thresh) or stick_down
        joy["P1 Left"]   = (buttons[3]  > btn_thresh) or stick_left
        joy["P1 Right"]  = (buttons[4]  > btn_thresh) or stick_right
        
        joy["P1 Start"]  = buttons[20] > btn_thresh 
        joy["P1 Select"] = buttons[1]  > btn_thresh 
        joy["P1 L"]      = buttons[8]  > btn_thresh 
        joy["P1 R"]      = buttons[15] > btn_thresh 
    elseif CONSOLE_TYPE == "NES" then
        joy["P1 A"]      = buttons[19] > btn_thresh 
        joy["P1 B"]      = buttons[6]  > btn_thresh 
        
        joy["P1 Up"]     = (buttons[5]  > btn_thresh) or stick_up
        joy["P1 Down"]   = (buttons[2]  > btn_thresh) or stick_down
        joy["P1 Left"]   = (buttons[3]  > btn_thresh) or stick_left
        joy["P1 Right"]  = (buttons[4]  > btn_thresh) or stick_right
        
        joy["P1 Start"]  = buttons[20] > btn_thresh 
        joy["P1 Select"] = buttons[1]  > btn_thresh 
    end
    
    joypad.set(joy)
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

-- Set timeout to 500ms so the emulator doesn't hang if the server is "thinking"
tcp.ReceiveTimeout = 500
tcp.SendTimeout = 500

console.log("Connected!")
local stream = tcp:GetStream()
local resp_buffer = luanet.import_type("System.Byte[]")(4096)

while tcp.Connected do
    -- 1. Screenshot (Now 'client' refers to BizHawk API correctly)
    client.screenshot(TEMP_IMG_FILE)
    
    -- 2. Read Bytes (Fast .NET read)
    local file_bytes = File.ReadAllBytes(TEMP_IMG_FILE)
    
    -- 3. Header
    local len = file_bytes.Length
    console.log("Sending " .. len .. " bytes")
    local json_header = string.format('{"type": "predict", "len": %d}\n', len)
    local header_bytes = Encoding.ASCII:GetBytes(json_header)
    
    -- 4. Send
    local send_ok, send_err = pcall(function()
        stream:Write(header_bytes, 0, header_bytes.Length)
        stream:Write(file_bytes, 0, file_bytes.Length)
    end)
    
    if not send_ok then
        console.log("Error sending data.")
        break
    end
    
    -- 5. Receive
    local read_ok, read_err = pcall(function()
        local bytes_read = stream:Read(resp_buffer, 0, resp_buffer.Length)
        if bytes_read > 0 then
            local resp_str = Encoding.ASCII:GetString(resp_buffer, 0, bytes_read)
            
            -- Extract raw arrays for buttons and sticks
            local s, e = string.find(resp_str, "\"buttons\":%s*%[")
            if e then
                local end_bracket = string.find(resp_str, "%]", e) -- Note: simplified parsing, assumes no nested brackets in numbers
                if end_bracket then
                    local buttons_str = string.sub(resp_str, e+1, end_bracket-1)
                    
                    -- Parse all button values into a flat list
                    local all_buttons = {}
                    for v in string.gmatch(buttons_str, "[%d%.]+") do
                        table.insert(all_buttons, tonumber(v))
                    end

                    -- Parse all stick values into a flat list
                    local all_sticks = {}
                    local sj, ej = string.find(resp_str, "\"j_left\":%s*%[")
                    if ej then
                        local sub_joy = string.sub(resp_str, ej + 1)
                        -- Try to find the closing bracket for j_left to limit search
                        local end_joy = string.find(sub_joy, "%]")
                        if end_joy then
                            sub_joy = string.sub(sub_joy, 1, end_joy)
                        end
                        
                        for v in string.gmatch(sub_joy, "[%-%d%.]+") do
                            table.insert(all_sticks, tonumber(v))
                        end
                    end
                    
                    -- Parse repeat count (default 1)
                    local repeat_count = 1
                    local s_rep, e_rep = string.find(resp_str, "\"repeat\":%s*%d+")
                    if e_rep then
                         local rep_str = string.match(string.sub(resp_str, s_rep, e_rep), "%d+")
                         repeat_count = tonumber(rep_str) or 1
                    end

                    -- Calculate number of frames based on buttons (21 buttons per frame)
                    -- We trust that stick values match in length (2 per frame)
                    local num_frames = math.floor(#all_buttons / 21)
                    
                    if num_frames > 0 then
                        console.log("Executing " .. num_frames .. " frames (x" .. repeat_count .. ")")
                        
                        for i = 0, num_frames - 1 do
                            -- Extract buttons for this frame
                            local frame_buttons = {}
                            for b = 1, 21 do
                                table.insert(frame_buttons, all_buttons[i * 21 + b])
                            end
                            
                            -- Extract stick for this frame (default 0)
                            local lx = 0
                            local ly = 0
                            if #all_sticks >= (i * 2 + 2) then
                                lx = all_sticks[i * 2 + 1]
                                ly = all_sticks[i * 2 + 2]
                            end
                            
                            -- Apply and Advance (Repeatedly)
                            for r = 1, repeat_count do
                                apply_controls(frame_buttons, lx, ly)
                                emu.frameadvance()
                            end
                            
                            -- Optional: Update GUI every frame or just once? 
                            -- Updating every frame might flicker or be slow, but accurate.
                            -- gui.drawText(0, 0, "AI Frame " .. (i+1) .. "/" .. num_frames, "green")
                        end
                    else
                        console.log("No valid frames parsed.")
                        emu.frameadvance() -- Fallback to advance at least once
                    end
                end
            else
                 emu.frameadvance()
            end
            
            gui.drawText(0, 0, "AI Active", "green")
        else
            emu.frameadvance()
        end
    end)

    if not read_ok then
        -- If timeout or error, just ignore and go to next frame
        console.log("Timeout/Skip: " .. tostring(read_err))
        emu.frameadvance()
    end

end

tcp:Close()
console.log("Disconnected.")