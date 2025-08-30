import numpy as np

from vision.ocr import OCR


def test_read_rewards_with_mocked_text(monkeypatch):
    o = OCR(enabled=False)
    def fake_read_text(_img):
        return ["100 Gold", "3 Cards"]
    monkeypatch.setattr(o, "read_text", fake_read_text)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    roi = (0.1, 0.1, 0.9, 0.9)
    res = o.read_rewards(img, roi)
    assert res["gold"] == 100
    assert res["cards"] == 3

