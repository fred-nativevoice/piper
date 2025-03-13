cat ../../etc/test_sentences/test_en-gb-x-rp.jsonl | \
    python3 -m piper_train.infer \
        --sample-rate 16000 \
        --checkpoint ./dataset_peppa_pig_v20250310_16khz/lightning_logs/version_0/checkpoints/epoch\=2490-step\=618560.ckpt \
        --output-dir peppa_sentences
