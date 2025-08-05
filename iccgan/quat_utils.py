import torch
import math


def quaternion_to_euler(quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Extract quaternion components
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Convert to Euler angles (roll, pitch, yaw)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # Check for gimbal lock
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.pi / 2,
            torch.asin(sinp)
        )
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=1)

def quatmultiply(q0, q1):
    x1, y1, z1, w1 = q0
    x2, y2, z2, w2 = q1
    return [
        w2*x1+x2*w1-y2*z1+z2*y1,
        w2*y1+x2*z1+y2*w1-z2*x1,
        w2*z1-x2*y1+y2*x1+z2*w1,
        w2*w1-x2*x1-y2*y1-z2*z1
    ]

def quatconj(q):
    return [-q[0], -q[1], -q[2], q[3]]

def quatdiff(q0, q1):
    return quatmultiply(q1, quatconj(q0))

def quatdiff_rel(q0, q1):
    return quatmultiply(quatconj(q0), q1)

def quat2axis_angle(q):
    x, y, z, w = q
    norm = (x*x + y*y + z*z + w*w)**0.5
    if norm == 0:
        return [0, 0, 1], 0
    
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    if abs(w) > 0.9999:
        return [0, 0, 1], 0
    
    angle = 2 * math.acos(w)
    
    sin_half_angle = math.sqrt(1 - w*w)
    axis = [x/sin_half_angle, y/sin_half_angle, z/sin_half_angle]
    
    return axis, angle
