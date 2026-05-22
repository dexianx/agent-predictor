# Dormant CatBoost Artifacts

CatBoost model/data artifacts were moved here after Pass 9 because the hidden-split checks did not justify using CatBoost as active Task A rating authority.

The large artifact folders are intentionally ignored by git:

- `dormant/catboost/data/`
- `dormant/catboost/models/`

Active Task A inference does not load these artifacts unless `USE_RATING_HEAD=1` and explicit `RATING_HEAD_*` paths are configured.
