# Running ActivitySim in Estimation Mode

ActivitySim can be run in estimation mode by including a few extra settings in
the `estimation.yaml` config file.  The key setting in this file is the `enabled`
setting, which must be set to `True` in order to run in estimation mode. The
default value for this setting is `False`, so if it is not explicitly set to
`True`, ActivitySim will run in normal simulation mode, and everything else
in the `estimation.yaml` config file will be ignored. These settings are
documented below. After running ActivitySim in estimation mode, the EDBs will be
written to disk, and can be used with Larch to re-estimate the model parameters.

```{eval-rst}
.. currentmodule:: activitysim.core.estimation
```

## Configuration Settings

```{eval-rst}
.. autopydantic_model:: EstimationConfig
    :inherited-members: PydanticReadable
    :show-inheritance:
```

## Survey Table Settings

```{eval-rst}
.. autopydantic_model:: SurveyTableConfig
```
