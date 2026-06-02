import xml.etree.ElementTree as ET

from selecta.export import export_cues
from selecta.features.types import PhraseRegion, TrackFeatures


def _track(path: str, title: str, artist: str, phrases: tuple[PhraseRegion, ...]) -> TrackFeatures:
    return TrackFeatures(
        path=path,
        duration=245.7,
        bpm=128.5,
        bpm_confidence=0.95,
        key_camelot="8A",
        key_confidence=0.88,
        phrases=phrases,
        title=title,
        artist=artist,
    )


def test_export_cues_writes_rekordbox_xml(tmp_path):
    tracks = [
        _track(
            path="fixtures/set-opener.mp3",
            title="Set Opener",
            artist="DJ Selecta",
            phrases=(
                PhraseRegion(start=0.0, end=32.0, bars=16, label="intro", confidence=0.92),
                PhraseRegion(start=32.0, end=64.0, bars=16, label="body", confidence=0.81),
                PhraseRegion(start=64.0, end=96.0, bars=16, label="drop", confidence=0.97),
                PhraseRegion(start=96.0, end=128.0, bars=16, label="breakdown", confidence=0.44),
                PhraseRegion(start=128.0, end=160.0, bars=16, label="outro", confidence=0.76),
            ),
        ),
        _track(
            path="fixtures/after-hours.mp3",
            title="After Hours",
            artist="Night Shift",
            phrases=(
                PhraseRegion(start=0.0, end=48.0, bars=24, label="body", confidence=0.63),
                PhraseRegion(start=48.0, end=96.0, bars=24, label="outro", confidence=0.39),
            ),
        ),
    ]
    out_path = tmp_path / "rekordbox.xml"

    export_cues(tracks, str(out_path))

    tree = ET.parse(out_path)
    root = tree.getroot()

    assert root.tag == "DJ_PLAYLISTS"

    collection = root.find("COLLECTION")
    assert collection is not None
    assert collection.attrib["Entries"] == str(len(tracks))

    track_elements = collection.findall("TRACK")
    assert len(track_elements) == len(tracks)

    first_track_marks = track_elements[0].findall("POSITION_MARK")
    assert len(first_track_marks) == len(tracks[0].phrases)

    for track_element in track_elements:
        assert track_element.attrib["Location"].startswith("file://localhost")
        for mark in track_element.findall("POSITION_MARK"):
            assert set(("Name", "Type", "Start", "Num")).issubset(mark.attrib)
            assert mark.attrib["Type"] == "0"
            if mark.attrib["Num"] != "-1":
                assert mark.attrib["Num"] in {str(index) for index in range(8)}

    assert first_track_marks[0].attrib["Name"] == "Intro"
    assert first_track_marks[0].attrib["Num"] == "0"
    assert first_track_marks[1].attrib["Num"] == "-1"
    assert first_track_marks[2].attrib["Num"] == "1"
    assert first_track_marks[3].attrib["Name"] == "Breakdown?"
    assert first_track_marks[3].attrib["Num"] == "2"
    assert first_track_marks[4].attrib["Num"] == "3"

    second_track_marks = track_elements[1].findall("POSITION_MARK")
    assert second_track_marks[0].attrib["Num"] == "-1"
    assert second_track_marks[1].attrib["Name"] == "Outro?"
    assert second_track_marks[1].attrib["Num"] == "0"
