def pre_patch_batch(images, patch_size):
    batch_size, channels, height, width = images.shape

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    # (batch_size, channels, n_p, n_p, p, p)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

    patches = patches.view(-1, channels, patch_size, patch_size)

    return patches

def patchify_labels(images, patch_size):
    batch_size, height, width = images.shape

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    patches = patches.contiguous().view(batch_size, -1, patch_size * patch_size)

    # take majority class per patch
    patch_labels = patches.mode(dim=2).values

    return patch_labels.reshape(-1)