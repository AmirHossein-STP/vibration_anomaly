"""Production wrapper around the three research pipelines.

One CLI (`python -m serving_core <service> <command>`), one folder contract,
one model registry -- shared by all three services. The science code stays in
`vibration_ad`, `hydraulic_fusion_ad` and `domain_adaptation`; this package
only adds what a hand-off needs: input validation, versioned models, a human
approval step, and result files in a fixed format. See DEPLOYMENT_PLAN.md.
"""
__version__ = "0.1.0"
