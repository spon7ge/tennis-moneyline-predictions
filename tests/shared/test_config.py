from tml.shared.config import get_settings


def test_settings_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PARLAY_API_KEY", "test-key-not-real")
    monkeypatch.setenv("TML_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    s = get_settings()
    assert s.parlay_api_key == "test-key-not-real"
    assert s.tml_data_dir == tmp_path
