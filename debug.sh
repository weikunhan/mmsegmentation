PROJECT_PATH=$(pwd)
if [[ "$PROJECT_PATH" == *"scripts"* ]]; then
  PROJECT_PATH=$(echo ${PROJECT_PATH%/*})
fi
cd $PROJECT_PATH
bash tools/dist_train.sh configs_ours/BraTS/deeplabv3_unet.py 1 --no-validate