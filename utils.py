import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import imageio
from PIL import Image as PILImage
import shutil
from IPython.display import Image, display


def uncompress_tar(tar_file_path, uncompressed_dir):
    if not os.path.isfile(tar_file_path):
        raise FileNotFoundError(f"Tar file not found: {tar_file_path}")

    if not os.path.exists(uncompressed_dir):
        os.makedirs(uncompressed_dir)

    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=uncompressed_dir)

    print(f"Uncompressed {tar_file_path} to {uncompressed_dir}")


def load_nifti_image(file_path):
    """Load a NIfTI file."""
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data


def create_gif_from_slices(img_data, axis=2, duration=0.1, resize_to=None):
    """
    Create a GIF from slices of a 3D image and display it.

    Parameters:
    img_data (numpy.ndarray): 3D image data.
    axis (int): Axis along which to slice the 3D image.
    duration (float): Duration between frames in the GIF.
    """
    # Get the number of slices along the specified axis
    num_slices = img_data.shape[axis]

    # Create a temporary directory to store the slice images
    temp_dir = tempfile.mkdtemp()

    # Create a list to store the filenames of the slice images
    slice_images = []

    for i in range(num_slices):
        plt.figure()
        if axis == 0:
            plt.imshow(img_data[i, :, :], cmap='gray')
        elif axis == 1:
            plt.imshow(img_data[:, i, :], cmap='gray')
        elif axis == 2:
            plt.imshow(img_data[:, :, i], cmap='gray')
        else:
            raise ValueError("Axis must be 0, 1, or 2.")
        plt.axis('off')

        # Save the slice image
        slice_filename = os.path.join(temp_dir, f"slice_{i:03d}.png")
        plt.savefig(slice_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Append the filename to the list
        slice_images.append(slice_filename)

    # Create the GIF
    gif_path = os.path.join(temp_dir, "output.gif")
    with imageio.get_writer(gif_path, mode='I', duration=duration, loop=0) as writer:
        for slice_filename in slice_images:
            image = imageio.v2.imread(slice_filename)
            if resize_to:
                image = PILImage.fromarray(image).resize(resize_to, PILImage.Resampling.LANCZOS)
                image = np.array(image)
            writer.append_data(image)

    # Display the GIF
    display(Image(filename=gif_path))

    # Clean up temporary slice images
    shutil.rmtree(temp_dir)
