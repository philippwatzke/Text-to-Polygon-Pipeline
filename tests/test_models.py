# tests/test_models.py
import pytest
from pydantic import ValidationError
from ki_geodaten.models import BBox, CreateJobRequest, TilePreset, ValidateBulkRequest, ValidationUpdate

def test_bbox_accepts_valid():
    b = BBox(minx=11.0, miny=48.0, maxx=11.1, maxy=48.1)
    assert b.as_tuple() == (11.0, 48.0, 11.1, 48.1)

def test_bbox_rejects_inverted():
    with pytest.raises(ValidationError):
        BBox(minx=11.1, miny=48.0, maxx=11.0, maxy=48.1)  # minx >= maxx
    with pytest.raises(ValidationError):
        BBox(minx=11.0, miny=48.1, maxx=11.1, maxy=48.0)  # miny >= maxy

def test_create_job_request_defaults_preset_to_medium():
    req = CreateJobRequest(prompt="building", bbox_wgs84=[11.0, 48.0, 11.1, 48.1])
    assert req.tile_preset == TilePreset.MEDIUM

def test_validate_bulk_request():
    r = ValidateBulkRequest(updates=[ValidationUpdate(pid=1, validation="REJECTED")])
    assert r.updates[0].pid == 1
    assert r.updates[0].validation == "REJECTED"

def test_validation_update_rejects_unknown_value():
    with pytest.raises(ValidationError):
        ValidationUpdate(pid=1, validation="MAYBE")
