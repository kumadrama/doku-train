# Existing Idea daily worker release

This directory supplies the standard CodeBuild/ECR and task-definition release manifest for the
existing `doku-prod-idea-mapped-daily` worker. Its family, IAM role, CPU/memory, log group, executor,
S3 bootstrap and two external EventBridge triggers are retained. There is no `task-target.yml`;
`dkd deploy apply svc` only registers the task revision. Trigger activation remains in the isolated
`doku-iac/envs/prod/idea_mapped_daily` stack after the current release state is idle.

The image inherits the exact runtime from revision 5, including Python 3.10.22 and AWS SDK
1.43.120. It copies the reviewed worker source and does not install or upgrade dependencies. Neither
FYP's `tf-pipeline` release configuration nor its task family is used for deployment.

## Prepare and verify

Run `python3 idea-mapped-daily/build/package_source.py --pin` after a runtime change. Review the
updated manifest, then run `idea-mapped-daily/build/build.sh`. The deterministic archive is written
to ignored `idea-mapped-daily/build/output/worker.tar.gz`; the image contains the identical source.
The ordinary build rejects an archive hash different from the deployment pin. The inference test
suite also verifies this pin, so a source-only change cannot silently reuse an old service build.

Upload the reviewed archive under its SHA-addressed existing Idea release prefix before registering
the task. Generate the binding from the actual IaC output and existing binding, preserving the
training/inference images and config; review only the new `idea_appconfig` resource wiring. Update
`DOKU_BINDING_URI` and `DOKU_BINDING_SHA256` together after its immutable S3 publication. The manifest pins the binding generated from the applied Idea resource outputs. Config content
remains unchanged. The matching immutable S3 object must exist before registration or activation.

## Standard publication

After review and merge to main, the normal commands are:

```sh
dkd ci init idea-mapped-daily --branch main
dkd ci build idea-mapped-daily --ref main
dkd deploy plan svc idea-mapped-daily --env prod --ref main --image <main-release-tag>
dkd deploy apply svc idea-mapped-daily --env prod --ref main --image <main-release-tag>
```

The first command provisions the per-service CodeBuild/ECR publication configuration for this
existing worker. It does not add a new training worker or scheduler. Review the deployment plan:
only `RegisterTaskDefinition`, original Idea family, no service or rule updates. After registry and
binding verification, the isolated IaC stack updates both existing Idea triggers to the new revision.
Do not change FYP triggers or invoke a training job to demonstrate a deployment.
