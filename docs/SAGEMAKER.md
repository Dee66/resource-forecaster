# SageMaker Serverless Deployment (portfolio guide)

This document describes how to package the model and deploy it to SageMaker Serverless for a portfolio demo.

1) Build an inference container
- Create a Dockerfile that installs your runtime (Python 3.11), dependencies (joblib, pandas, scikit-learn), and copies the `sagemaker_inference.py` entrypoints into `/opt/ml/code/`.

Example (Dockerfile snippet):

```
FROM public.ecr.aws/lambda/python:3.11
RUN pip install joblib pandas scikit-learn
COPY scripts/sagemaker_inference.py /opt/ml/code/
COPY serve /usr/local/bin/serve  # optional custom serve script
ENTRYPOINT ["/usr/local/bin/serve"]
```

2) Package the model
- Use `scripts/package_forecaster_artifacts.py` to create a zip/tarball of the model artifact containing `model.joblib` or similar.

3) Push image to ECR and upload artifact to S3
- Push the built image to your account ECR and note the image URI.
- Upload the model artifact to S3; the demo `deploy_sagemaker_serverless.py` script can also upload it for you.

4) Deploy (demo)
- Run the example deploy script:

```bash
python scripts/deploy_sagemaker_serverless.py \
  --s3-bucket my-bucket --s3-prefix models/forecaster \
  --model-artifact ./artifacts/model_package-20251007-150642.zip \
  --image-uri 123456789012.dkr.ecr.us-east-1.amazonaws.com/forecaster-inference:latest \
  --endpoint-name rf-portfolio-demo --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole --region us-east-1 --invoke
```

5) Cleanup
- To remove resources created by the demo script, use `--cleanup`.

Notes & Limitations
- This repo includes example scripts and templates for demonstration purposes only — review IAM, networking, and cost considerations before running in a shared account.
