# ASR using NVIDIA NeMo for Telugu
NVIDIA NeMo [stt_en_fastconformer_ctc_large](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large) finetuned on open-source Telugu data.

Data sources:

* https://ai4bharat.iitm.ac.in/shrutilipi/
* https://www.openslr.org/66/
* https://huggingface.co/datasets/google/fleurs/viewer/te_in - train
* https://www.iitm.ac.in/donlab/indictts/database

(plus white noise, background noise, pitch-shifted augmentations)

Performance:
0.21 CER on Google FLEURS Test data

You can find the model and the inference interface at https://huggingface.co/spaces/bharathraj-v/fastconformer_ctc_telugu
