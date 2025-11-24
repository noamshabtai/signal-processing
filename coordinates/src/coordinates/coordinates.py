import numpy as np


def spherical_to_ned(R, theta_deg, phi_deg):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    x_north = R * np.cos(theta) * np.cos(phi)
    y_east = R * np.cos(theta) * np.sin(phi)
    z_down = -R * np.sin(theta)

    return np.array([x_north, y_east, z_down])
