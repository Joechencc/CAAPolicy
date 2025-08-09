import math
import carla

def _wrap180(a): return (a + 180.0) % 360.0 - 180.0
def _clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def _signed_speed(vehicle: carla.Vehicle):
    tf = vehicle.get_transform()
    vyaw = math.radians(tf.rotation.yaw)
    fx, fy = math.cos(vyaw), math.sin(vyaw)
    v = vehicle.get_velocity()
    return v.x*fx + v.y*fy  # m/s, signed along forward

class PID1D:
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0
        self.prev = 0.0
        self.first = True
        self.out_min, self.out_max = out_min, out_max
    def step(self, e, dt):
        if dt <= 0: return 0.0
        self.i += e * dt
        d = 0.0 if self.first else (e - self.prev) / dt
        self.first = False
        self.prev = e
        u = self.kp*e + self.ki*self.i + self.kd*d
        return _clamp(u, self.out_min, self.out_max)

class YawPositionPIDController:
    """
    Lateral: PID on lateral error + PID on yaw error
    Longitudinal: PID on signed speed (reverse-aware)
    """
    def __init__(self, vehicle: carla.Vehicle,
                 # lateral position (cross-track) PID
                 lat_kp=1.2, lat_ki=0.0, lat_kd=0.1,
                 # yaw PID
                 yaw_kp=1.0, yaw_ki=0.0, yaw_kd=0.05,
                 # longitudinal PID
                 lon_kp=0.8, lon_ki=0.0, lon_kd=0.05,
                 max_throttle=0.6, max_brake=0.8, max_steer=0.8,
                 reverse_speed_scale=0.8,
                 steer_rate_limit=0.12):
        self._veh = vehicle
        self.lat_pid = PID1D(lat_kp, lat_ki, lat_kd, out_min=-1.0, out_max=1.0)
        self.yaw_pid = PID1D(yaw_kp, yaw_ki, yaw_kd, out_min=-1.0, out_max=1.0)
        self.lon_pid = PID1D(lon_kp, lon_ki, lon_kd, out_min=-1.0, out_max=1.0)

        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.max_steer = max_steer
        self.reverse_speed_scale = reverse_speed_scale
        self.steer_rate_limit = steer_rate_limit
        self.prev_steer = 0.0

    @staticmethod
    def _to_transform(target):
        # accept carla.Waypoint (has .transform) or carla.Transform directly
        return target.transform if hasattr(target, "transform") else target

    def _signed_speed(self, tf: carla.Transform):
        vel = self._veh.get_velocity()
        vyaw = math.radians(tf.rotation.yaw)
        fx, fy = math.cos(vyaw), math.sin(vyaw)
        return vel.x * fx + vel.y * fy

    def _errors_lateral_yaw(self, cur_tf: carla.Transform, tgt_tf: carla.Transform):
        """
        Compute lateral error in the vehicle frame (meters) and yaw error (radians).
        Lateral error sign: left positive (using vehicle forward/right axes).
        """
        # vehicle axes in world
        vyaw = math.radians(cur_tf.rotation.yaw)
        fx, fy = math.cos(vyaw), math.sin(vyaw)          # forward unit (world)
        rx, ry = -math.sin(vyaw), math.cos(vyaw)         # right unit (world)

        dx = tgt_tf.location.x - cur_tf.location.x
        dy = tgt_tf.location.y - cur_tf.location.y

        # signed lateral error: projection on right axis
        lat_err = dx * rx + dy * ry  # >0 means target is to the right

        # yaw error (radians, wrapped)
        yaw_err_deg = _wrap180(tgt_tf.rotation.yaw - cur_tf.rotation.yaw)
        yaw_err = math.radians(yaw_err_deg)

        return lat_err, yaw_err

    def run_step(self, target_speed, waypoint, dt=0.05):
        """
        Reverse-safe PID:
        - target_speed: desired *forward* speed magnitude (m/s), e.g. 4.0
        - waypoint: Waypoint-like (has .transform) or a carla.Transform
        """
        # --- normalize target transform ---
        tgt_tf = waypoint.transform if hasattr(waypoint, "transform") else waypoint
        assert isinstance(tgt_tf, carla.Transform)

        veh_tf  = self._veh.get_transform()
        veh_loc = veh_tf.location
        veh_yaw = veh_tf.rotation.yaw

        # --- decide forward vs reverse by projection ---
        vyaw = math.radians(veh_yaw)
        fx, fy = math.cos(vyaw), math.sin(vyaw)
        dx = tgt_tf.location.x - veh_loc.x
        dy = tgt_tf.location.y - veh_loc.y
        proj = dx*fx + dy*fy          # >0 ahead, <0 behind
        reverse = (proj < 0.0)

        # -------- LATERAL (steering) --------
        # your existing lateral PID that outputs steer in [-1, 1]:
        steer = self._lat_controller.run_step(tgt_tf)
        if reverse:
            steer = -steer  # backing up inverts steering effect

        # steering rate limit & clamp (your original code)
        if steer > self.past_steering + 0.1:
            steer = self.past_steering + 0.1
        elif steer < self.past_steering - 0.1:
            steer = self.past_steering - 0.1
        steer = _clamp(steer, -self.max_steer, self.max_steer)

        # -------- LONGITUDINAL (speed magnitude PID) --------
        v_signed = _signed_speed(self._veh)   # m/s, signed along forward
        v_mag    = abs(v_signed)                  # use magnitude
        v_ref    = float(target_speed)
        if reverse:
            # optional: gentler in reverse
            v_ref *= getattr(self, "reverse_speed_scale", 0.6)

        # PID on |speed|, not signed speed
        # self._lon_mag_pid should be PID1D(kp,ki,kd, out_min=0, out_max=1)
        acc_cmd = self._lon_mag_pid.step(v_ref - v_mag, dt)  # 0..1 roughly

        # Decide throttle vs brake based on whether we need to speed up or slow down
        throttle = 0.0
        brake = 0.0
        speed_tol = 0.1  # m/s deadband
        if v_mag < v_ref - speed_tol:
            # need to speed up → throttle
            throttle = _clamp(acc_cmd, 0.0, self.max_throt)
            brake = 0.0
        elif v_mag > v_ref + speed_tol:
            # need to slow down → brake
            throttle = 0.0
            # scale brake with how much we exceed the target
            over = v_mag - v_ref
            brake = _clamp(over / max(1.0, v_ref + 1e-6), 0.0, self.max_brake)
        else:
            throttle = 0.0
            brake = 0.0

        # -------- Build control --------
        ctrl = carla.VehicleControl()
        ctrl.throttle = float(throttle)
        ctrl.brake    = float(brake)
        ctrl.steer    = float(steer)
        ctrl.hand_brake = False
        # EITHER:
        ctrl.manual_gear_shift = False
        ctrl.reverse = bool(reverse)
        # OR (older CARLA): 
        # ctrl.manual_gear_shift = True
        # ctrl.gear = -1 if reverse else 1

        self.past_steering = ctrl.steer
        return ctrl