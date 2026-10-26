
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut


def load_dicom_image(path, img_size, voi_lut=True, rotate=0):
    try:
        dicom = pydicom.dcmread(path)
        
        # Apply VOI LUT transformation if requested and available
        if voi_lut:
            try:
                data = apply_voi_lut(dicom.pixel_array, dicom)
            except Exception:
                data = dicom.pixel_array
        else:
            data = dicom.pixel_array
            
        # Standardize pixel output data type
        data = data.astype(np.float32)
        
        # Apply rotation if specified
        if rotate != 0:
            # Rotates matrix in 90-degree increments
            k = rotate // 90
            data = np.rot90(data, k=k)
            
    except Exception:
        return np.zeros((img_size, img_size), dtype=np.float32)
        
    return data



def load_dicom_images_3d(files, num_imgs, img_size, rotate=0, data_directory="."):
    if len(files) == 0:
        return np.zeros((num_imgs, img_size, img_size), dtype=np.float32)

    middle = len(files) // 2
    half = num_imgs // 2
    p1, p2 = max(0, middle - half), min(len(files), middle + half)

    imgs = []
    for f in files[p1:p2]:
        im = load_dicom_image(f, img_size=img_size, rotate=rotate, voi_lut=True)
        im = np.nan_to_num(np.asarray(im, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if im.ndim != 2:                       # multi-frame / RGB files
            im = np.zeros((img_size, img_size), dtype=np.float32)
        imgs.append(im)

    if not imgs:
        return np.zeros((num_imgs, img_size, img_size), dtype=np.float32)

    # use the most common slice shape; replace slices with another shape by zeros
    shapes = [im.shape for im in imgs]
    target = max(set(shapes), key=shapes.count)
    imgs = [im if im.shape == target else np.zeros(target, dtype=np.float32) for im in imgs]
    return np.stack(imgs, axis=0)

