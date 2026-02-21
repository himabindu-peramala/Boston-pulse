## Description

<!-- Provide a clear and concise description of your changes -->

### Dataset-Specific (if implementing a new dataset)

<!-- Complete all if adding a new dataset -->

- [ ] Created `data-pipeline/src/datasets/{dataset}/ingest.py` with watermark support
- [ ] Created `data-pipeline/src/datasets/{dataset}/preprocess.py` with all transformations
- [ ] Created `data-pipeline/src/datasets/{dataset}/features.py` with documented features
- [ ] Created `data-pipeline/schemas/{dataset}/raw_schema.json`
- [ ] Created `data-pipeline/schemas/{dataset}/processed_schema.json`
- [ ] Created `data-pipeline/schemas/{dataset}/features_schema.json`
- [ ] Created `data-pipeline/configs/datasets/{dataset}.yaml` with API endpoint and schedule
- [ ] Created `data-pipeline/dags/datasets/{dataset}_dag.py` with all validation stages
- [ ] Added bias slice configurations
- [ ] Created `data-pipeline/src/datasets/{dataset}/README.md` with dataset documentation
- [ ] Added tests in `data-pipeline/tests/unit/datasets/{dataset}/`

### Schema Changes (if modifying schemas)

- [ ] Schema changes are backwards compatible OR migration plan is documented
- [ ] Schema tests have been added/updated
- [ ] TFDV statistics will be regenerated
