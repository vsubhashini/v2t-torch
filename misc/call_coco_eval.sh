#!/bin/bash

cd coco_evaluation
python write_to_json.py $1 -x
python run_coco_eval.py $2 youtube_val.json
cd ../
