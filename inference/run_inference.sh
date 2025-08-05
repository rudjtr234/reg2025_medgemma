#!/bin/bash

conda activate medgemma

# 디렉토리 이동
cd /home/mts/ssd_16tb/member/jks/reg2025_medgemma/inference

# 추론 실행
python answer_to_make_json_v2.py
