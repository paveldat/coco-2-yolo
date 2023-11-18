## COCO2YOLO
COCO to YOLO converter.
Given the annotation JSON file, this tool will help you create TXT files for YOLO learning.
The TXT file has the same name as the image from which it was created.

## Example of use
```bash
python3 coco2yolo.py -j labels.json -o output
```

## Example of the created TXT file
```txt
67 0.281460 0.071347 0.222480 0.119120
48 0.453370 0.506373 0.674140 0.770773
```
