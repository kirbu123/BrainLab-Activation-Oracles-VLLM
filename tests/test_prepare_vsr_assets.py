import json
from pathlib import Path

from scripts.prepare_vsr_assets import prepare_images


def test_prepare_vsr_assets_links_existing_coco_file(tmp_path: Path):
    coco_dir = tmp_path / "coco"
    coco_dir.mkdir()
    source = coco_dir / "000000000001.jpg"
    source.write_bytes(b"jpeg")

    annotations = tmp_path / "train.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "image": source.name,
                "image_link": "https://example.invalid/unused.jpg",
                "caption": "A test.",
                "label": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "vsr-assets"

    prepare_images([annotations], [coco_dir], output_dir)

    linked = output_dir / source.name
    assert linked.is_symlink()
    assert linked.resolve() == source.resolve()
