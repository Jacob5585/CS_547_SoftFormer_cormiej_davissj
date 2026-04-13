mkdir dataset/openearthmap-sar
curl https://zenodo.org/records/14622048/files/dfc25_track1_trainval.zip?download=1 -o ./dataset/openearthmap-sar/download.zip
cd ./dataset/openearthmap-sar/
unzip download.zip