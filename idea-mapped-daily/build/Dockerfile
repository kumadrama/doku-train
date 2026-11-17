# Same validated Idea runtime; only the reviewed worker source is refreshed.
FROM 703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/doku-reco-train@sha256:13db53a9f6a4b03753f9f3c1efc5a5c605bef82151eed7687fd7c3fb6296e74a
COPY build/output/source/recommend_tf /opt/idea-worker/recommend_tf
WORKDIR /opt/idea-worker
ENV PYTHONPATH=/opt/idea-worker
RUN python -m compileall -q recommend_tf && python -c "import recommend_tf.platforms.sagemaker.executor, recommend_tf.platforms.sagemaker.idea_publication; from botocore.session import Session; assert 'LatestDeploymentNumber' in Session().get_service_model('appconfig').operation_model('StartDeployment').input_shape.members"
