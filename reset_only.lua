local PARAM_TABLE_KEY = 90
local PARAM_TABLE_PREFIX = "SIM_ERES_"

assert(param:add_table(PARAM_TABLE_KEY, PARAM_TABLE_PREFIX, 2), "failed to create param table")

function bind_add_param(name, idx, default_value)
    assert(param:add_param(PARAM_TABLE_KEY, idx, name, default_value))
    return Parameter(PARAM_TABLE_PREFIX .. name)
end

local P_RESET = bind_add_param("RESET", 1, 0)
local P_STATE = bind_add_param("STATE", 2, 0)

-- Core params
param:set("ARMING_CHECK", 0)
param:set("GPS_TYPE", 0)
param:set("GPS_AUTO_CONFIG", 0)
param:set("GPS_AUTO_SWITCH", 0)
param:set("AHRS_EKF_TYPE", 10)

-- Disable ALL failsafes at startup
param:set("BATT_FS_LOW_ACT", 0)
param:set("BATT_FS_CRT_ACT", 0)
param:set("BATT_LOW_VOLT", 0)
param:set("BATT_CRT_VOLT", 0)
param:set("BATT_ARM_VOLT", 0)
param:set("FS_CRASH_CHECK", 0)
param:set("MOT_SPIN_MIN", 0.15)
param:set("FS_DR_ENABLE", 0)
param:set("FS_VIBE_ENABLE", 0)
param:set("FS_EKF_ACTION", 0)
param:set("FS_GCS_ENABLE", 0)
param:set("FS_THR_ENABLE", 0)
param:set("SIM_BATT_VOLTAGE", 12.6)
param:set("SIM_BATT_CAP_AH", 200)
param:set("MOT_THST_HOVER", 0.35)
param:set("MOT_HOVER_LEARN", 0)
param:set("EK3_AFFINITY", 0)
param:set("EK3_IMU_MASK", 1)

local home_location = nil
local base_attitude = Quaternion()
local home_saved = false

local function ready() return ahrs:get_origin() end
local function vec3(x,y,z) local v=Vector3f() v:x(x);v:y(y);v:z(z); return v end

function update()
    if not home_saved then
        if ready() then
            home_location = ahrs:get_location()
            base_attitude = ahrs:get_quaternion()
            home_saved = true
            gcs:send_text(0, "Home locked at startup!")
        end
        return
    end

    if P_RESET:get() == 1 then
        P_RESET:set(0)
        sim:set_pose(0,
            home_location,
            base_attitude,
            vec3(0,0,0),
            vec3(0,0,0))
        if arming:is_armed() then arming:disarm() end
        param:set("SIM_BATT_VOLTAGE", 12.6)
        param:set("SIM_BATT_CAP_AH", 200)
        param:set("MOT_THST_HOVER", 0.35)
        param:set("MOT_HOVER_LEARN", 0)
        param:set("EK3_AFFINITY", 0)
        param:set("EK3_IMU_MASK", 1)
        battery:reset_remaining(0, 100.0)
        gcs:send_text(0, "Teleported and reset!")
        P_STATE:set(1)
    end
end

function loop()
    update()
    return loop, 50
end

gcs:send_text(0, "Reset script loaded!")
return loop, 1000
