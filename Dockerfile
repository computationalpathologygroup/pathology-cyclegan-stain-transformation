FROM dodrio1.umcn.nl:uokbaseimage/diag:tf2.8-pt1.10-v1
LABEL authors="linda"

ENV SM_FRAMEWORK=tf.keras
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir -U \
    image-classifiers \
    albumentations==1.2.0 \
    segmentation-models \
    tensorflow_probability==0.11.1

WORKDIR /home/user/source

COPY libs/pathology-common ./pathology-common
COPY libs/pathology-fast-inference ./pathology-fast-inference
COPY libs/diag-models ./diag-models
COPY libs/pathology-cyclegan-stain-transformation ./pathology-cyclegan-stain-transformation

ENV PYTHONPATH="/home/user/source/pathology-common:/home/user/source/diag-models:/home/user/source/pathology-fast-inference:/home/user/source/pathology-cyclegan-stain-transformation:${PYTHONPATH}"

CMD ["python3"]

ENTRYPOINT []