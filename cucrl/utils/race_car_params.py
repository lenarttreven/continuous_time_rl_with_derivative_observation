from flax import struct


@struct.dataclass
class CarParams:
    m: float = 0.05
    l: float = 0.06
    a: float = 0.25
    b: float = 0.01
    g: float = 9.81
    d_f: float = 0.2
    c_f: float = 1.25
    b_f: float = 2.5
    d_r: float = 0.2
    c_r: float = 1.25
    b_r: float = 2.5
    c_m_1: float = 0.2
    c_m_2: float = 0.05
    c_rr: float = 0.0
    c_d_max: float = 0.1
    c_d_min: float = 0.01
    tv_p: float = 0.0
    q_pos: float = 0.1
    q_v: float = 0.1
    r_u: float = 1
    room_boundary: float = 80.0
    velocity_limit: float = 100.0
    max_steering: float = 0.25
    dt: float = 0.01
    control_freq: int = 5

    def _get_x_com(self):
        x_com = self.l * (self.a + 2) / (3 * (self.a + 1))
        return x_com

    def _get_moment_of_intertia(self):
        # Moment of inertia around origin
        a = self.a
        b = self.b
        m = self.m
        l = self.l
        i_o = m / (6 * (1 + a)) * ((a ** 3 + a ** 2 + a + 1) * (b ** 2) + (l ** 2) * (a + 3))
        x_com = self._get_x_com()
        i_com = i_o - self.m * (x_com ** 2)
        return i_com
