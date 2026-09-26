import json, pathlib, jsonschema, pytest
R = pathlib.Path(__file__).resolve().parents[1]
S = json.loads((R / "contracts/analysis.schema.json").read_text())
def test_schema_is_valid(): jsonschema.Draft202012Validator.check_schema(S)
def test_fixture_validates(): jsonschema.validate(json.loads((R / "tests/fixtures/analysis_fake.json").read_text()), S)
def test_bad_label_rejected():
    d = json.loads((R / "tests/fixtures/analysis_fake.json").read_text()); d["verdict"]["label"] = "MAYBE"
    with pytest.raises(jsonschema.ValidationError): jsonschema.validate(d, S)
def test_unknown_field_rejected():
    d = json.loads((R / "tests/fixtures/analysis_fake.json").read_text()); d["surprise"] = 1
    with pytest.raises(jsonschema.ValidationError): jsonschema.validate(d, S)
