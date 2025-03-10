#!/usr/bin/bash
ulimit -v 209715200 # 200 GB = 200 * 1024 * 1024 KB = 209715200 KB

# python data/verify_and_download.py
python data/utils.py