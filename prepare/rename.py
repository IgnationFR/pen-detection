import glob
import os
import random


def rename_files(path_to_files: str, path_to_save: str, name_prefix="image"):
    """ Function to find, rename and move .jpg images in specific path.

    Args:
        path_to_files: Path to search images.
        path_to_save: Path to save renamed images.
        name_prefix: New name prefix of the images.

    """
    files = glob.glob("%s/*" % path_to_files)
    random.shuffle(files)
    for i in range(0, len(files)):
        os.rename(files[i], os.path.join(path_to_save, name_prefix + "-" + str(i) + ".jpg"))
    print("Images renamed and moved to %s" % path_to_save)


if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), 'dataset/images')
    new_image_path = os.path.join(os.getcwd(), 'dataset/renamed-images')
    rename_files(image_path, new_image_path, name_prefix="image")
