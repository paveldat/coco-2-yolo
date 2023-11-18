"""
        ██▄██ ▄▀▄ █▀▄ █▀▀ . █▀▄ █░█
        █░▀░█ █▄█ █░█ █▀▀ . █▀▄ ▀█▀
        ▀░░░▀ ▀░▀ ▀▀░ ▀▀▀ . ▀▀░ ░▀░
▒▐█▀█─░▄█▀▄─▒▐▌▒▐▌░▐█▀▀▒██░░░░▐█▀█▄─░▄█▀▄─▒█▀█▀█
▒▐█▄█░▐█▄▄▐█░▒█▒█░░▐█▀▀▒██░░░░▐█▌▐█░▐█▄▄▐█░░▒█░░
▒▐█░░░▐█─░▐█░▒▀▄▀░░▐█▄▄▒██▄▄█░▐█▄█▀░▐█─░▐█░▒▄█▄░
"""

import json
import argparse
from pathlib import Path


class Coco2Yolo:
    """
    Class to convert COCO to YOLO.
    """

    def __init__(self, json_file_path: [str, Path],
                 output_folder: [str, Path]) -> None:
        """
        Constructor.

        Args:
            * json_file_path - Path to JSON file
            * output_folder - Path to output folder
        """

        self.__json_file_path = Path(json_file_path)
        self.__output_folder = Path(output_folder)
        self.__check_json_and_dir()
        self.__labels = self.__get_labels()
        self.__coco_id_name_map = self.__categories()
        self.__coco_name_list = list(self.__coco_id_name_map.values())
        self.__print_info()

    def __print_info(self):
        """
        Prints number of images, categories, annotations.
        """

        labels_keys = ['images', 'categories', 'annotations']
        for key in labels_keys:
            if key in self.__labels.keys():
                print(f'Number of {key}: {len(self.__labels[key])}')

    def __check_json_and_dir(self) -> None:
        """
        Checks if JSON file exists otherwise raise error.
        Checks if output folder exists otherwise create it.
        """

        print(f'Checking if {self.__json_file_path} exists')
        if not self.__json_file_path.exists:
            raise ValueError(f'{self.__json_file_path} not found')
        print(f'Checking if {self.__output_folder} exists')
        self.__output_folder.mkdir(exist_ok=True, parents=True)

    def __get_labels(self) -> dict:
        """
        Reads JSON file and returns labels.
        """

        print('Get labels from JSON file')
        with open(self.__json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def __categories(self) -> dict:
        """
        Returns all categories from JSON file.
        """

        print('Get categories from labels')
        categories = {}
        for cls in self.__labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def __load_images_info(self) -> dict:
        """
        Loads images info.

        Returns:
            * Dictionary: id: (filename, width, height)
        """

        print('Load images info')
        images_info = {}
        for image in self.__labels['images']:
            image_id = image['id']
            filename = image['file_name']
            width = image["width"]
            height = image["height"]
            images_info[image_id] = (filename, width, height)
        return images_info

    def __convert_bbox(self, bbox: dict, width: int, height: int) -> tuple:
        """
        Converts bbox.

        Args:
            * bbox - Bbox annotation
            * width - Image width
            * height - Image height

        Returns:
            Center x and y, width and height
        """

        pos_x, pos_y, pos_w, pos_h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = pos_x + pos_w / 2
        centery = pos_y + pos_h / 2
        centerx *= (1 / width)
        centery *= (1 / height)
        pos_w *= (1 / width)
        pos_h *= (1 / height)
        return centerx, centery, pos_w, pos_h

    def __convert_annotations(self, images_info: dict) -> dict:
        """
        Converts annotations.

        Args:
            * images_info - dictionary of the image id, filename,
                            width, height
        """

        print('Convert annotations')
        annotations_dict = {}
        if 'annotations' not in self.__labels.keys():
            raise ValueError("Annotations not found in labels")
        for annotations in self.__labels["annotations"]:
            bbox = annotations["bbox"]
            img_id = annotations["image_id"]
            category_id = annotations["category_id"]

            img_info = images_info.get(img_id)
            img_name = img_info[0]
            img_w = img_info[1]
            img_h = img_info[2]
            yolo_box = self.__convert_bbox(bbox, img_w, img_h)

            annotation_info = (img_name, category_id, yolo_box)
            annotation_infos = annotations_dict.get(img_id)
            if not annotation_infos:
                annotations_dict[img_id] = [annotation_info]
            else:
                annotation_infos.append(annotation_info)
                annotations_dict[img_id] = annotation_infos
        return annotations_dict

    def __save_txt_file(self, annotations: dict) -> None:
        """
        Saves anootations to txt file.

        Args:
            * annotations - Annotations dictionary
        """

        print('Save txt files')
        for _, value in annotations.items():
            filename = self.__output_folder / f'{Path(value[0][0]).stem}.txt'
            with open(filename, 'w', encoding='utf-8') as file:
                for obj in value:
                    category_name = self.__coco_id_name_map.get(obj[1])
                    category_id = self.__coco_name_list.index(category_name)
                    box = ' '.join(['{:.6f}'.format(el) for el in obj[2]])
                    line = f'{category_id} {box}'
                    file.write(f'{line}\n')

    def convert(self):
        """
        Converter.
        """

        images_info = self.__load_images_info()
        annotations = self.__convert_annotations(images_info)
        self.__save_txt_file(annotations)

    @staticmethod
    def coco2yolo(json_file_path: [str, Path], output_path: [str, Path]):
        """
        Converts COCO to YOLO.

        Args:
            * json_file_path - Path to JSON file
            * output_folder - Path to output folder
        """

        Coco2Yolo(json_file_path, output_path).convert()


def main():
    parser = argparse.ArgumentParser(description='Coco to YOLO converter.')
    parser.add_argument('-j', '--json-path', help='Path to JSON file',
                        dest='json', required=True)
    parser.add_argument('-o', '--output-path', help='Path to output folder',
                        dest='out', required=True)

    known_args, _ = parser.parse_known_args()
    Coco2Yolo.coco2yolo(known_args.json, known_args.out)


if __name__ == '__main__':
    main()
