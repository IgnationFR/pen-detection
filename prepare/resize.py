from PIL import Image
import glob
import os


def resize_images(path_to_images, max_width=680):
    """ Resize image above a certain width.

    Args:
        path_to_images: Path to the images
        max_width: The size above which you need to reduce it.

    """
    images = glob.glob("%s/*.jpg" % path_to_images)
    for image in images:
        img = Image.open(image)
        width, height = img.size
        if width > max_width:
            new_width = max_width
            new_height = int(new_width * height / width)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            img.save(image, format="JPEG")


if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), 'dataset/images')
    new_image_path = os.path.join(os.getcwd(), 'dataset/renamed-images')
    resize_images("dataset/images")