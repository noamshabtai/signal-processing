import numpy as np


def spherical_to_ned(r, az, el):
    z = -r * (np.sin(el))
    A = r * (np.cos(el))
    y = A * (np.sin(az))
    x = A * (np.cos(az))
    return x, y, z


def ned_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arcsin(-z / r)
    az = np.arctan2(y, x)
    return r, az, el


def spherical_to_enu(r, az, inc):
    z = r * (np.cos(inc))
    A = r * (np.sin(inc))
    y = A * (np.sin(az))
    x = A * (np.cos(az))
    return x, y, z


def enu_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    inc = np.arccos(z / r)
    A = r * (np.sin(inc))
    az = np.arccos(x / A) * np.sign(y)
    return r, az, inc


def distance_to(v):
    return np.linalg.norm(v, axis=-1, keepdims=0)
