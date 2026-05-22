import importlib
import warnings
import numpy as np
from PIL import Image
from typing import Tuple, Union

import os, requests, zipfile

def open_tiff_file(name: str) -> np.ndarray:
    img = Image.open(name)

    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))

    return np.array(frames).squeeze()

def download_and_unzip(url, extract_to, chain_path):
    """Download and unzip using a verified SSL certificate."""
    if os.path.exists(extract_to):
        print(f"The directory '{extract_to}' already exists. Skipping download and extraction.")
        return

    local_zip = os.path.basename(url)
    print(f"📥 Downloading {local_zip} (approx 1min)...")

    try:
        response = requests.get(url, stream=True, verify=chain_path, timeout=20)
        response.raise_for_status()
        with open(local_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ {local_zip} downloaded successfully ({os.path.getsize(local_zip)} bytes)")

        print(f"📂 Extracting to '{extract_to}/'...")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(local_zip)
        print(f"✅ Extraction completed to '{extract_to}'")

    except requests.exceptions.RequestException as e:
        print(f"❌ Download error: {e}")
    except zipfile.BadZipFile:
        print("❌ The downloaded file is not a valid ZIP archive.")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")


class OTF:
    """
    The Optical Transfer Function of the optical system.
    Generates a 2-D image from an exponential approximation of the ideal OTF.
    """

    def __init__(self, na: float, wavelength: float, pixel_size: float, image_size: int, curvature: float):
        """
        :param na: Numerical aperture of the optical system.
        :param wavelength: Wavelength of the emitted light in nanometers.
        :param pixel_size: Physical size of the image pixel in micrometers.
        :param image_size: Width and height of the image.
        :param curvature: Bend of the model OTF function in [0, 1] range where 1 is a perfect OTF.
        :param samples: Number of values to sample.
        """
        cutoff_frequency = 1000 * 2 * na / wavelength  # in micrometer^-1 (for incoherent imaging)
        self.image_cutoff = cutoff_frequency * pixel_size * image_size
        self.image_size = image_size
        self.curvature = curvature

    def __call__(self, size: int = None, x_shift: float = 0, y_shift: float = 0) -> np.ndarray:
        """
        Generate a 2-D image representation.
        :param size: Width and height of the generated image.
        :param x_shift: Sub-pixel shift along the x-axis of the OTF center.
        :param y_shift:Sub-pixel shift along the y-axis of the OTF center.
        :return: 2-D representation of the OTF
        """
        if size is None:
            size = self.image_size

        x, y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]),
                           np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]))
        distance_to_origin = np.hypot(x + x_shift, y + y_shift)

        return self.value(np.minimum(distance_to_origin / self.image_cutoff, 1))

    def double(self, x_shift: float = 0, y_shift: float = 0) -> np.ndarray:
        return self(self.image_size * 2, x_shift, y_shift)

    def value(self, x):
        return (2 / np.pi) * (np.arccos(x) - x * np.sqrt(1 - x * x)) * self.curvature ** x

def apodization_filter(dist_ratio: float, bend: float, size: int) -> np.ndarray:
    """
    Generate an apodization filter that can be directly multiplied with the summed fft result image.

    :param dist_ratio: A coefficient that transform the point distance in pixels to a value where [0, 1] range
        corresponds to the extended OTF support.
    :param bend: A coefficient in the [0, 1] range defining how much medium frequencies should be augmented.
    :param size: Size of the apodization filter.
    :return: np.ndarray of size (size, size) of the apodization filter.
    """
    x, y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]),
                       np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]))

    distance = np.hypot(x, y) * dist_ratio
    mask = np.bitwise_and(0 <= distance, distance < 1)

    apo = np.power(
        (2 / np.pi) * (np.arccos(distance, where=mask) - distance * np.sqrt(1 - distance * distance, where=mask)),
        bend,
        where=mask
    )

    return np.where(mask, apo, 0)


def wiener_filter(shifts, otf: OTF, w: float, size: int) -> np.ndarray:
    """
    Generate a wiener filter denominator so that it can be directly multiplied with the summed fft result image.

    :param shifts: A container of length 3 of shifts of the second component for each of the 3 orientations.
    :param otf: OTF object.
    :param w: Wiener parameter.
    :param size: Size of the wiener filter.
    :return: np.ndarray of size (size, size) of the Wiener filter denominator.
    """
    otf0 = np.abs(otf(size)) ** 2
    wiener = 3 * otf0

    for i in range(3):
        otf1 = np.abs(otf(size, shifts[i][1], shifts[i][0])) ** 2
        otf2 = np.abs(otf(size, -shifts[i][1], -shifts[i][0])) ** 2

        wiener += otf1 + otf2

    return 1 / (wiener + w * w)

def map_otf_support(components: np.ndarray, shifts, otf: OTF) -> None:
    """
    Multiply conjugated OTF to corresponding parts of it's support in the separated and shifted components. Values
    outside the OTF support are set to zero. This function alters the original images.

    :param components: np.ndarray of shape (9, size, size) of padded separated and shifted components in frequency space.
    :param shifts: A container of length 3 of shifts of the second component for each of the 3 orientations.
    :param otf: OTF object
    """
    size = components.shape[-1]

    components[::3] *= np.conjugate(otf(size))
    for i in range(3):
        components[i * 3 + 1] *= np.conjugate(otf(size, shifts[i][1], shifts[i][0]))
        components[i * 3 + 2] *= np.conjugate(otf(size, -shifts[i][1], -shifts[i][0]))


def fourier_shift(fft_image: np.ndarray, x_shift: Union[float, np.ndarray], y_shift: Union[float, np.ndarray]) -> np.ndarray:
    """
    Use the fourier shift theorem to perform a sub-pixel shift of a frequency image in the spatial domain.

    It is implemented in a way that allows the x_shift and y_shift parameters to be a np.ndarray of arbitrary shape with
    the last two axes being np.newaxis or None. The returned array shape is then x/y_shift.shape + (height, width)

    :param fft_image: np.ndarray of shape (height, width).
    :param x_shift: Sub-pixel shift along the x-axis.
    :param y_shift: Sub-pixel shift along the y-axis.
    :return: np.ndarray of shape (height, width) of the shifted frequency image.
    """
    height, width = fft_image.shape[-2:]

    spatial_image = np.fft.ifft2(fft_image)

    x_indices = np.arange(-width // 2, width // 2, dtype=int)
    y_indices = np.arange(-height // 2, height // 2, dtype=int)

    x = np.exp(-1j * 2 * np.pi * x_shift * x_indices[None, :] / width)
    y = np.exp(-1j * 2 * np.pi * y_shift * y_indices[:, None] / height)

    shifted_spatial_image = spatial_image * (x * y)

    return np.fft.fft2(shifted_spatial_image)


def shift_components(components: np.ndarray, shifts) -> None:
    """
    Use the fourier shift theorem to perform a sub-pixel shift of frequency components in the spatial domain. This
    function alters the original images.

    :param components: np.ndarray of shape (9, height, width) of padded separated components in frequency space.
    :param shifts: A container of length 3 of shifts of the second component for each of the 3 orientations.
    :return: np.ndarray of shape (9, height, width) of shifted components.
    """
    for i in range(3):
        components[i * 3 + 1, ...] = fourier_shift(components[i * 3 + 1, ...], shifts[i][1], shifts[i][0])
        components[i * 3 + 2, ...] = fourier_shift(components[i * 3 + 2, ...], -shifts[i][1], -shifts[i][0])


def pad_components(components: np.ndarray) -> np.ndarray:
    """
    Pad an image to twice its size from the center, keeping the corners intact.
    This makes it possible to pad images that are NOT centered using np.fft.fftshift.

    :param components: np.ndarray of shape (..., size, size) of separated components in frequency space.
    :return: np.ndarray of shape (..., size, size) padded with zeros from the center.
    """
    size = components.shape[-1]
    padded_components = np.zeros(components.shape[:-2] + (size * 2, size * 2), dtype=np.complex128)

    x, y = np.meshgrid(
        np.hstack([np.arange(size // 2), np.arange(size * 3 // 2, size * 2)]),
        np.hstack([np.arange(size // 2), np.arange(size * 3 // 2, size * 2)])
    )

    padded_components[..., y, x] = components

    return padded_components

def _component_separation_matrix(phase_offset: float = 0, modulation: float = 1) -> np.ndarray:
    phases = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3]) + phase_offset

    M = np.ones((3, 3), dtype=np.complex128)


    M[:, 1] = 0.5 * modulation * (np.cos(phases) + 1j * np.sin(phases))
    M[:, 2] = 0.5 * modulation * (np.cos(-phases) + 1j * np.sin(-phases))

    return np.linalg.inv(M)

def separate_components(fft_images: np.ndarray, phase_offsets=np.zeros(3), phase_modulations=np.ones(3)) -> np.ndarray:
    """
    Compute the spectral separation of components.

    :param fft_images: np.ndarray of shape (9, height, width) of images in frequency space.
    :param phase_offsets: A container of length 3 of phase offsets for each of the 3 orientations, zeros by default.
    :param phase_modulations: A container of length 3 of phase modulations for each of the 3 orientations,
        ones by default.
    :return: np.ndarray of shape (9, height, width) of separated components.
    """
    separation_matrices = np.array([_component_separation_matrix(phase_offsets[i], phase_modulations[i])
                                    for i in range(3)])

    return np.einsum('aij,ajkl->aikl',separation_matrices,
                     fft_images.reshape(3, 3, *fft_images.shape[1:])).reshape(9, *fft_images.shape[1:])


def run_reconstruction(fft_images: np.ndarray, otf: OTF, shifts, phase_offsets, modulations, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a SIM reconstruction using the provided parameters.

    :param fft_images: np.ndarray of shape (9, height, width) of images in frequency space.
    :param otf: OTF object
    :param shifts: A container of length 3 of shifts of the second component for each of the 3 orientations.
    :param phase_offsets: A container of length 3 of phase offsets for each of the 3 orientations.
    :param modulations: A container of length 3 of phase modulations for each of the 3 orientations.
    :param config: A dictionary of SIM reconstruction parameters. It must contain the following keys: ["wavelength",
        "na", "px_size", "apo_cutoff", "wiener_parameter", "apo_bend"]
    :return: fft_result, spatial_result
    """
    size = fft_images.shape[-1]
    cutoff = 1000 * 2 * config["na"] / config["wavelength"]
    apo_dist_ratio = 1 / (config["px_size"] * config["apo_cutoff"] * cutoff * size)

    components = separate_components(fft_images, phase_offsets, modulations)
    components = pad_components(components)

    shift_components(components, shifts)
    map_otf_support(components, shifts, otf)

    wiener = wiener_filter(shifts, otf, config["wiener_parameter"], size * 2)
    apodization = apodization_filter(apo_dist_ratio, config["apo_bend"], size * 2)

    fft_result = np.sum(components, 0) * wiener * apodization
    spatial_result = np.real(np.fft.ifft2(fft_result))

    return fft_result, spatial_result

def illumination_pattern(angle, frequency, phase_offset, amplitude, size) -> np.ndarray:
    n = size // 2
    Y, X = np.mgrid[-n:n, -n:n]
    ky, kx = np.sin(angle) * frequency, np.cos(angle) * frequency
    return 1 + amplitude * np.cos(2 * np.pi * (X * kx + Y * ky) + phase_offset)


class PerlinNoise:
    """Adapted from: https://github.com/pvigier/perlin-numpy"""

    def __init__(self, size: int, res: int):
        """
        :param size: length of both dimensions of the generated noise image
        :param res: number of periods of noise to generate along each axis
        """

        meshgrid = np.mgrid[0:res:res / size, 0:res:res / size]
        self.grid = np.stack(meshgrid) % 1

        self.t = self._fade(self.grid)
        self.d = size // res
        self.sample_size = (res + 1, res + 1)

    def __call__(self) -> np.array:
        """
        Returned values are always in the [0, 1] range with ~0.5 mean
        """
        angles = 2 * np.pi * np.random.random_sample(self.sample_size)
        gradients = np.dstack((np.cos(angles), np.sin(angles))).repeat(self.d, 0).repeat(self.d, 1)

        n00 = (np.dstack((self.grid[0], self.grid[1])) * gradients[:-self.d, :-self.d]).sum(2)
        n10 = (np.dstack((self.grid[0] - 1, self.grid[1])) * gradients[self.d:, :-self.d]).sum(2)
        n01 = (np.dstack((self.grid[0], self.grid[1] - 1)) * gradients[:-self.d, self.d:]).sum(2)
        n11 = (np.dstack((self.grid[0] - 1, self.grid[1] - 1)) * gradients[self.d:, self.d:]).sum(2)

        n0 = n00 * (1 - self.t[0]) + n10 * self.t[0]
        n1 = n01 * (1 - self.t[0]) + n11 * self.t[0]

        return 0.5 + 2 ** -0.5 * (n0 * (1 - self.t[1]) + n1 * self.t[1])

    @staticmethod
    def _fade(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def _require_import(module_name: str, package_name: str = None):
    if package_name is None:
        package_name = module_name

    try:
        return importlib.import_module(module_name)
    except ImportError:
        warnings.warn(
            f"Optional IO dependency '{package_name}' is missing. "
            "Install it manually or with `pip install shape2fate[io]`.")
        raise


def _open_mrc_file(name: str) -> np.ndarray:
    mrcfile = _require_import("mrcfile")

    with warnings.catch_warnings():
        # mrcfile can emit warnings for slightly malformed headers; ignore them here
        warnings.simplefilter("ignore")
        with mrcfile.open(name, permissive=True) as mrc:
            return np.flip(mrc.data, -2)  # Flip y-axis because mrc origin is in the bottom left


def _open_tiff_file(name: str) -> np.ndarray:
    Image = _require_import("PIL.Image", "Pillow (PIL)")

    img = Image.open(name)

    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))

    return np.array(frames).squeeze()


def _open_czi_file(name: str) -> np.ndarray:
    czifile = _require_import("czifile")
    return czifile.imread(name).squeeze()


def _open_nd2_file(name: str) -> np.ndarray:
    nd2 = _require_import("nd2")
    return nd2.imread(name)


def open_image_file(name: str) -> np.ndarray:
    _, ext = os.path.splitext(name)

    match ext.lower():
        case '.tiff' | '.tif':
            return _open_tiff_file(name)
        case '.mrc' | '.dv' | '.otf':
            return _open_mrc_file(name)
        case '.czi':
            return _open_czi_file(name)
        case '.nd2':
            return _open_nd2_file(name)
        case _:
            raise ValueError(f'"{ext}" is an unrecognized file extension')
