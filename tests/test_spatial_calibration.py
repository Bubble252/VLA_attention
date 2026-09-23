import numpy as np

from scripts.calibrate_spatial_metrics import normalize, resize_bilinear, sample_metrics


def test_resize_preserves_shape_and_mass_after_normalization():
    source = np.arange(16, dtype=np.float64).reshape(4, 4)
    result = normalize(resize_bilinear(source, 8, 8))
    assert result.shape == (8, 8)
    assert np.isclose(result.sum(), 1.0)


def test_calibration_reports_soft_and_threshold_metrics():
    grid = np.zeros((8, 8), dtype=np.float64)
    grid[2:4, 3:5] = 1.0
    metrics = sample_metrics(grid / grid.sum(), [(30, 20, 50, 40)], 80, 80, [0.2], [0.8])
    assert 0.0 <= metrics["soft_iou"] <= 1.0
    assert "box_iou_top20" in metrics
    assert "box_iou_q80" in metrics
