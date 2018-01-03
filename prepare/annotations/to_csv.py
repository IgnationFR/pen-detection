import glob
import xml.etree.ElementTree as ElementTree
import os
import pandas as pd


def xml_annotations_to_data_frame(path_to_annotations):
    """ Translate the annotations from xml files

    Args:
        path_to_annotations: Path where the xml annotations files are.

    Returns:
        pandas.DataFrame: The translation of the annotations.

    """
    xml_list = []
    for xml_file in glob.glob(path_to_annotations + "/*.xml"):
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            value = (root.find("filename").text,
                     int(root.find("size")[0].text),
                     int(root.find("size")[1].text),
                     member[0].text,
                     int(member[5][0].text),
                     int(member[5][1].text),
                     int(member[5][2].text),
                     int(member[5][3].text)
                     )
            xml_list.append(value)

    column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == "__main__":
    image_path = "../../dataset/renamed-images/annotations"
    xml_df = xml_annotations_to_data_frame(image_path)
    xml_df.to_csv("../../dataset/annotations.csv", index=None)
    print("Successfully converted xml to csv.")
