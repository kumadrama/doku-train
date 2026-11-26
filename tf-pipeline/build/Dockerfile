# Reuse the verified CPU OS/runtime, install the current hash-locked dependencies,
# and replace the complete source tree from this main-channel build.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY requirements.txt /tmp/managed-requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/managed-requirements.txt
RUN rm -rf /workspace/recommend_tf
COPY recommend_tf /workspace/recommend_tf
WORKDIR /workspace
ENTRYPOINT []
CMD ["python", "-m", "recommend_tf.platforms.sagemaker.executor"]
