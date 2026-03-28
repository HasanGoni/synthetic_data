"""
Epic 5 — Labeling session management for human-in-the-loop annotation.

Manages the workflow of preparing image batches for annotation, tracking
progress, and loading completed labels back into the training pipeline.

Folder layout::

    <output_dir>/<session_name>/
        images/           # copies of selected images
        masks/            # annotator places label PNGs here (empty placeholders created)
        selected.csv      # copy of the selection CSV

Typical usage::

    from udm_epic5.labeling.session import create_labeling_session, load_labeled_samples

    session = create_labeling_session(
        selected_csv="outputs/selection/round_001.csv",
        images_dir="/data/target/images",
        output_dir="outputs/labeling",
    )
    session.prepare()
    print(session.status())

    # ... human annotates masks in the session folder ...

    image_paths, mask_paths = load_labeled_samples(str(session.session_dir))
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Supported image file extensions (mirrors udm_epic4 convention).
_IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".bmp"}


class LabelingSession:
    """Manage a single labeling session for human-in-the-loop annotation.

    A session consists of a set of images selected by an active learning
    strategy and a folder structure where annotators place binary masks.

    Parameters
    ----------
    selected_csv : str
        CSV file listing selected images.  Must contain a column ``image``
        (or ``filename``, ``image_path``, ``file``) with image file names
        or paths.
    images_dir : str
        Root directory containing the original target-domain images.
    output_dir : str
        Base directory for labeling sessions.
    session_name : str
        Subdirectory name for this session (default ``"session_001"``).
    """

    def __init__(
        self,
        selected_csv: str,
        images_dir: str,
        output_dir: str,
        session_name: str = "session_001",
    ) -> None:
        self.selected_csv = Path(selected_csv)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.session_name = session_name

        self.session_dir = self.output_dir / self.session_name
        self.images_out = self.session_dir / "images"
        self.masks_out = self.session_dir / "masks"

        # Create directory structure
        self.images_out.mkdir(parents=True, exist_ok=True)
        self.masks_out.mkdir(parents=True, exist_ok=True)

        # Load selected image list
        self._selected_df = self._load_selection()

    def _load_selection(self) -> pd.DataFrame:
        """Read the selection CSV and normalise the filename column.

        Supports CSV files with columns named ``image``, ``filename``,
        ``image_path``, or ``file``.  The resolved column is stored as
        ``image`` for consistency.

        Returns:
            DataFrame with at least an ``image`` column.
        """
        df = pd.read_csv(self.selected_csv)

        # Normalise the image column name
        name_candidates = ["image", "filename", "image_path", "file"]
        image_col = None
        for col in name_candidates:
            if col in df.columns:
                image_col = col
                break

        if image_col is None:
            raise ValueError(
                f"Selection CSV must contain one of {name_candidates}. "
                f"Found columns: {list(df.columns)}"
            )

        if image_col != "image":
            df = df.rename(columns={image_col: "image"})

        logger.info(
            "Loaded %d selected images from %s",
            len(df),
            self.selected_csv,
        )
        return df

    def _resolve_image_path(self, image_ref: str) -> Path:
        """Resolve an image reference to a full path on disk.

        If the reference is already an absolute path that exists, use it
        directly.  Otherwise, treat it as a filename relative to
        :attr:`images_dir`.
        """
        ref_path = Path(image_ref)
        if ref_path.is_absolute() and ref_path.exists():
            return ref_path
        candidate = self.images_dir / ref_path.name
        if candidate.exists():
            return candidate
        # Try the raw reference as a relative path
        candidate = self.images_dir / image_ref
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Cannot locate image '{image_ref}' in {self.images_dir}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Copy selected images to the session folder and create empty mask placeholders.

        For each selected image, the image is copied to ``{session}/images/``
        and an empty file is created at ``{session}/masks/{stem}.png`` as a
        placeholder for the annotator.  A copy of the selection CSV is also
        placed in the session directory.
        """
        # Copy selection CSV for reference
        shutil.copy2(
            str(self.selected_csv),
            str(self.session_dir / "selected.csv"),
        )

        n_copied = 0
        for _, row in self._selected_df.iterrows():
            image_ref = str(row["image"])
            try:
                src_path = self._resolve_image_path(image_ref)
            except FileNotFoundError:
                logger.warning("Skipping missing image: %s", image_ref)
                continue

            dst_image = self.images_out / src_path.name
            if not dst_image.exists():
                shutil.copy2(src_path, dst_image)

            # Create empty mask placeholder (annotator replaces with real mask)
            mask_placeholder = self.masks_out / f"{src_path.stem}.png"
            if not mask_placeholder.exists():
                mask_placeholder.touch()

            n_copied += 1

        logger.info(
            "Prepared session '%s': %d images copied to %s",
            self.session_name,
            n_copied,
            self.session_dir,
        )

    def status(self) -> dict:
        """Return a summary of labeling progress.

        A mask is considered *completed* if the corresponding file in the
        ``masks/`` directory exists and is non-empty (size > 0 bytes).

        Returns:
            Dictionary with keys ``total``, ``labeled``, ``unlabeled``,
            ``progress_pct``.
        """
        total = 0
        labeled = 0

        for img_path in sorted(self.images_out.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            total += 1
            mask_path = self.masks_out / f"{img_path.stem}.png"
            if mask_path.exists() and mask_path.stat().st_size > 0:
                labeled += 1

        unlabeled = total - labeled
        progress_pct = (labeled / total * 100.0) if total > 0 else 0.0

        return {
            "total": total,
            "labeled": labeled,
            "unlabeled": unlabeled,
            "progress_pct": round(progress_pct, 1),
        }

    def summary(self) -> pd.DataFrame:
        """Return per-image labeling status as a DataFrame.

        Returns:
            DataFrame with columns ``image``, ``mask_exists``,
            ``mask_size_bytes``, ``labeled``.
        """
        rows: list[dict] = []
        for img_path in sorted(self.images_out.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue

            mask_path = self.masks_out / f"{img_path.stem}.png"
            mask_exists = mask_path.exists()
            mask_size = mask_path.stat().st_size if mask_exists else 0
            is_labeled = mask_exists and mask_size > 0

            rows.append(
                {
                    "image": img_path.name,
                    "mask_exists": mask_exists,
                    "mask_size_bytes": mask_size,
                    "labeled": is_labeled,
                }
            )

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        stat = self.status()
        return (
            f"LabelingSession(session='{self.session_name}', "
            f"labeled={stat['labeled']}/{stat['total']})"
        )


# ------------------------------------------------------------------
# Factory & utility functions
# ------------------------------------------------------------------


def create_labeling_session(
    selected_csv: str,
    images_dir: str,
    output_dir: str,
) -> LabelingSession:
    """Create a new :class:`LabelingSession` with an auto-generated session name.

    Inspects *output_dir* for existing ``session_NNN`` directories and
    increments the counter for the new session.

    Args:
        selected_csv: Path to the CSV listing selected images.
        images_dir:   Directory containing original target images.
        output_dir:   Base directory for labeling sessions.

    Returns:
        A new :class:`LabelingSession` instance (directory structure created).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-increment session name
    existing = sorted(
        d.name
        for d in output_path.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    )
    if existing:
        last_num = int(existing[-1].split("_")[-1])
        session_name = f"session_{last_num + 1:03d}"
    else:
        session_name = "session_001"

    session = LabelingSession(
        selected_csv=selected_csv,
        images_dir=images_dir,
        output_dir=output_dir,
        session_name=session_name,
    )
    return session


def load_labeled_samples(session_dir: str) -> Tuple[List[str], List[str]]:
    """Load completed labels from a labeling session directory.

    Scans the ``images/`` and ``masks/`` subdirectories of *session_dir*
    and returns paths only for images whose mask file is non-empty.

    Args:
        session_dir: Path to the session directory (e.g.
                     ``outputs/labeling/session_001``).

    Returns:
        Tuple of ``(image_paths, mask_paths)`` where each list contains
        absolute path strings for completed annotations only.
    """
    session_path = Path(session_dir)
    images_dir = session_path / "images"
    masks_dir = session_path / "masks"

    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise NotADirectoryError(f"Masks directory not found: {masks_dir}")

    image_paths: List[str] = []
    mask_paths: List[str] = []

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        mask_path = masks_dir / f"{img_path.stem}.png"
        if mask_path.exists() and mask_path.stat().st_size > 0:
            image_paths.append(str(img_path.resolve()))
            mask_paths.append(str(mask_path.resolve()))

    logger.info(
        "Loaded %d labeled samples from %s",
        len(image_paths),
        session_dir,
    )
    return image_paths, mask_paths
