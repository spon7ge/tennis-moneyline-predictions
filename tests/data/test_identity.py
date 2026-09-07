from tml.data.identity import PlayerIdentityMap


def test_identity_map_version_tagged() -> None:
    identity_map = PlayerIdentityMap(version="v1")
    assert identity_map.resolve("123") == "123"
    assert identity_map.version == "v1"
