import monai.transforms as mt

def get_base_transform(num_slices, image_size, mri_keys):
    # Always runs: resize + normalize. Needed for every sample, aug or not.
    return mt.Compose([
        mt.EnsureChannelFirstd(keys=mri_keys, channel_dim="no_channel"),
        mt.Resized(keys=mri_keys, spatial_size=(num_slices, image_size, image_size), mode="trilinear"),
        mt.NormalizeIntensityd(keys=mri_keys, nonzero=True, channel_wise=True),
    ])
    
def get_rand_transform(mri_keys):
    # Only runs for training when aug=True. No resize/normalize here — base already did it.
    return mt.Compose([
        mt.RandFlipd(keys=mri_keys, prob=0.5, spatial_axis=2),
        mt.RandFlipd(keys=mri_keys, prob=0.5, spatial_axis=1),
        mt.RandRotated(
            keys=mri_keys,
            range_x=0.2, range_y=0.2, range_z=0.2,
            prob=0.5, mode="bilinear", padding_mode="zeros"
        ),
        mt.RandZoomd(keys=mri_keys, min_zoom=0.9, max_zoom=1.1, prob=0.5),
        mt.RandGaussianNoised(keys=mri_keys, prob=0.3, mean=0.0, std=0.05),
        mt.RandShiftIntensityd(keys=mri_keys, offsets=0.1, prob=0.4),
        mt.RandScaleIntensityd(keys=mri_keys, factors=0.15, prob=0.4),
        mt.RandBiasFieldd(keys=mri_keys, prob=0.3, coeff_range=(0.0, 0.3)),
        mt.RandGibbsNoised(keys=mri_keys, prob=0.15, alpha=(0.0, 0.4)),
    ])
