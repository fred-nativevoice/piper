python3 -m piper_train.preprocess \
  --language en \
  --input-dir ../dataset_peppa/dataset_peppa_pig_v20250310_16khz/ \
  --output-dir ./dataset_peppa_pig_v20250310_16khz \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 16000 \
  --phoneme-type text \
  --text-casing lower 
