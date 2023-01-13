#!/bin/bash

# Usage: sh sample/make_voc_cocojson.sh

# echo $(find "$PWD/" -name "*.xml") | tr " " "\n" > annpaths_list.txt

# echo $(ls JPEGImages/*.jpg | sed s/"JPEGImages\/"// | sed s/"\.jpg"// ) | tr " " "\n" > img_list.txt

# grep -ERoh '<name>(.*)</name>' "Annotations" | sort | uniq | sed 's/<name>//g' | sed 's/<\/name>//g' > labels.txt

for split in train val trainval test
do
    python voc2coco.py \
        --ann_dir Annotations \
        --ann_ids annpaths_list.txt \
        --labels labels.txt \
        --output out.json
done