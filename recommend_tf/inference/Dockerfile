# Supply a verified CPU tensorflow/serving image pinned by sha256 digest.
ARG TF_SERVING_IMAGE
FROM ${TF_SERVING_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENV PYTHONPATH=/workspace PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY recommend_tf/inference /workspace/recommend_tf/inference
EXPOSE 8080
ENTRYPOINT ["python3", "-m", "recommend_tf.inference.app"]
CMD ["serve"]
