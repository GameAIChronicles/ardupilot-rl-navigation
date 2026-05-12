import time
import numpy as np
import matplotlib.pyplot as plt
from pymavlink import mavutil

DEVICE = 'tcp:127.0.0.1:5762'
ACTION_REPEAT = 4   # 🔥 change this to test (3,4,5)

master = mavutil.mavlink_connection(DEVICE)
master.wait_heartbeat()
print("Connected")

master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    20,
    1
)

def get_fresh_msg(msg_type, timeout=0.5):
    msg = None
    while True:
        new_msg = master.recv_match(type=msg_type, blocking=False)
        if new_msg is None:
            break
        msg = new_msg
    if msg is None:
        msg = master.recv_match(type=msg_type, blocking=True, timeout=timeout)
    return msg

def get_pos():
    msg = get_fresh_msg('LOCAL_POSITION_NED')
    if msg:
        return msg.x, msg.y, -msg.z, msg.vx, msg.vy, -msg.vz
    return 0,0,0,0,0,0

def set_guided():
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4
    )
    time.sleep(1)

def arm():
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1,0,0,0,0,0,0
    )
    time.sleep(2)

def takeoff(alt):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,0,0,0,0,0,0,alt
    )
    while True:
        _, _, z, *_ = get_pos()
        if z >= alt * 0.85:
            break
        time.sleep(0.2)

def send_velocity(vx, vy):
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0,0,0,
        vx, vy, 0,
        0,0,0,
        0,0
    )

# --- Setup ---
set_guided()
arm()
takeoff(10)

STEP_DURATION = 3.0
INNER_DT = 0.05

actions = [
    (0,0),
    (2,0),
    (0,0),
    (-2,0),
    (0,0)
]

log = []
start_global = time.time()

for vx_cmd, vy_cmd in actions:
    print(f"Action vx={vx_cmd}")

    start = time.time()
    while time.time() - start < STEP_DURATION:

        # 🔥 ACTION REPEAT BLOCK
        for _ in range(ACTION_REPEAT):
            send_velocity(vx_cmd, vy_cmd)
            time.sleep(INNER_DT)

        # observe AFTER repeats
        north, east, alt, vx, vy, vz = get_pos()

        t = time.time() - start_global
        log.append([t, vx_cmd, vy_cmd, vx, vy])

log = np.array(log)

t = log[:,0]
vx_cmd = log[:,1]
vx_real = log[:,3]

# --- Delay estimation ---
corr = np.correlate(vx_real - vx_real.mean(), vx_cmd - vx_cmd.mean(), mode='full')
lag = corr.argmax() - (len(vx_cmd) - 1)

# 🔥 important: effective dt = INNER_DT * ACTION_REPEAT
effective_dt = INNER_DT * ACTION_REPEAT
delay_sec = lag * effective_dt

print(f"\nEffective delay with ACTION_REPEAT={ACTION_REPEAT}: {delay_sec:.3f} sec")

# --- Plot ---
plt.figure()
plt.plot(t, vx_cmd, label="Command vx")
plt.plot(t, vx_real, label="Actual vx")

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title(f"With Action Repeat={ACTION_REPEAT} | Delay ≈ {delay_sec:.2f}s")

plt.legend()
plt.grid()

plt.savefig(f"delay_repeat_{ACTION_REPEAT}.png")
plt.show()