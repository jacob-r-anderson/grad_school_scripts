import numpy as np
import einops
import starfile

from yet_another_imod_wrapper.utils.io import read_xf, read_tlt


XF_FILE = 'TS_07.xf'
TLT_FILE = 'TS_07.tlt'
TILT_IMAGE_DIMENSIONS = (3838, 3710)  # (x, y)
TOMOGRAM_DIMENSIONS = (3838, 3710, 1000)  # (x, y, z)
IMPORT_STAR_FILE = '../../ImportTomo/job003/tomograms.star'
TOMO_NAME = 'TS_07'  # whichever tomo name you gave the xf and tlt files for
OUTPUT_STAR_FILE = 'alister_output.star'

# get alignment info from file
xf = read_xf(XF_FILE)
tlt = read_tlt(TLT_FILE)


inverse_rotation_matrices = xf[:, :4].reshape((-1, 2, 2)).transpose((0, 2, 1))
xf_shifts = xf[:, -2:].reshape((-1, 2, 1))
specimen_shifts = inverse_rotation_matrices @ -xf_shifts
specimen_shifts -= 0.5  # correct for IMOD rotation center -> relion rotation center
specimen_shifts = specimen_shifts.reshape((-1, 2))
in_plane_rotations = np.rad2deg(np.arccos(xf[:, 0]))

def Rx(angles_degrees: np.ndarray) -> np.ndarray:
    """Affine matrix for a rotation around the X-axis."""
    angles_degrees = np.asarray(angles_degrees).reshape(-1)
    c = np.cos(np.deg2rad(angles_degrees))
    s = np.sin(np.deg2rad(angles_degrees))
    matrices = einops.repeat(
        np.eye(4), 'i j -> n i j', n=len(angles_degrees)
    )
    matrices[:, 1, 1] = c
    matrices[:, 1, 2] = -s
    matrices[:, 2, 1] = s
    matrices[:, 2, 2] = c
    return np.squeeze(matrices)


def Ry(angles_degrees: np.ndarray) -> np.ndarray:
    """Affine matrix for a rotation around the Y-axis."""
    angles_degrees = np.asarray(angles_degrees).reshape(-1)
    c = np.cos(np.deg2rad(angles_degrees))
    s = np.sin(np.deg2rad(angles_degrees))
    matrices = einops.repeat(
        np.eye(4), 'i j -> n i j', n=len(angles_degrees)
    )
    matrices[:, 0, 0] = c
    matrices[:, 0, 2] = s
    matrices[:, 2, 0] = -s
    matrices[:, 2, 2] = c
    return np.squeeze(matrices)


def Rz(angles_degrees: float) -> np.ndarray:
    """Affine matrix for a rotation around the Z-axis."""
    angle_degrees = np.asarray(angles_degrees).reshape(-1)
    c = np.cos(np.deg2rad(angle_degrees))
    s = np.sin(np.deg2rad(angle_degrees))
    matrices = einops.repeat(
        np.eye(4), 'i j -> n i j', n=len(angle_degrees)
    )
    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c
    return np.squeeze(matrices)


def S(shifts: np.ndarray) -> np.ndarray:
    """Affine matrices for shifts.

    Shifts supplied can be 2D or 3D.
    """
    shifts = np.asarray(shifts, dtype=float)
    if shifts.shape[-1] == 2:
        shifts = _promote_2d_to_3d(shifts)
    shifts = np.array(shifts).reshape((-1, 3))
    matrices = einops.repeat(np.eye(4), 'i j -> n i j', n=shifts.shape[0])
    matrices[:, 0:3, 3] = shifts
    return np.squeeze(matrices)


def _promote_2d_to_3d(shifts: np.ndarray) -> np.ndarray:
    """Promote 2D vectors to 3D with zeros in the last dimension."""
    shifts = np.asarray(shifts).reshape(-1, 2)
    shifts = np.c_[shifts, np.zeros(shifts.shape[0])]
    return np.squeeze(shifts)


def get_relion_matrices(
        in_plane_rotations,
        tilt_angles,
        specimen_shifts,
        tilt_image_dimensions,
        tomogram_dimensions
):
    #tilt_image_center = tilt_image_dimensions / 2  # rln rotation center
    #specimen_center = tomogram_dimensions / 2

    tilt_image_center = tuple(x/2 for x in tilt_image_dimensions)  # rln rotation center
    specimen_center = tuple(x/2 for x in tomogram_dimensions)


    # Transformations, defined in order of application
    specimen_center=np.asarray(specimen_center)
    #print(specimen_center)

    #exit()
    s0 = S(-specimen_center)  # put specimen center-of-rotation at the origin
    # r0 = Rx(euler_angles['rlnTomoXTilt'])  # rotate specimen around X-axis
    r1 = Ry(tilt_angles)  # rotate specimen around Y-axis
    r2 = Rz(in_plane_rotations)  # rotate specimen around Z-axis
    s1 = S(specimen_shifts)  # shift projected specimen in xy (camera) plane
    s2 = S(tilt_image_center)  # move specimen back into tilt-image coordinate system

    # compose matrices
    # transformations = s2 @ s1 @ r2 @ r1 @ r0 @ s0
    transformations = s2 @ s1 @ r2 @ r1 @ s0
    return np.squeeze(transformations)


# calculate projection matrices
matrices = get_relion_matrices(
    in_plane_rotations,
    tlt,
    specimen_shifts,
    TILT_IMAGE_DIMENSIONS,
    TOMOGRAM_DIMENSIONS,
)

# put into from projection matrices into star file
star = starfile.read(IMPORT_STAR_FILE)

projection_matrix_labels = [f'rlnTomoProj{ax}' for ax in 'XYZW']
for idx, label in enumerate(projection_matrix_labels):
    rows = matrices[:, idx, :]
    star[TOMO_NAME][label] = [
        f'[{r[0]:.13g},{r[1]:.13g},{r[2]:.13g},{r[3]:.13g}]'
        for r in rows
    ]

starfile.write(star, OUTPUT_STAR_FILE)
