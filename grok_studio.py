"""
Grok Studio - Image & Video Generation GUI
Uses xAI's Grok API to generate images and videos.
Prompt rewriting: text model creates variations before image generation.
"""

import os
import sys
import json
import time
import base64
import subprocess
import threading
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox, QSpinBox,
    QScrollArea, QFrame, QFileDialog, QMessageBox, QSlider,
    QProgressBar, QStackedWidget, QSplitter, QGroupBox, QCheckBox,
    QSizePolicy, QToolButton, QLayout, QLayoutItem,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QUrl, QSize, QRect, QPoint,
)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont, QPalette, QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
VIDEOS_DIR = OUTPUT_DIR / "videos"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("XAI_API_KEY", "")
BASE_URL = "https://api.x.ai/v1"

# Text model for prompt rewriting
TEXT_MODEL = "grok-3-mini-fast"


# ---------------------------------------------------------------------------
# Helper: download a file from URL
# ---------------------------------------------------------------------------
def download_file(url: str, dest: Path) -> Path:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


# ---------------------------------------------------------------------------
# Worker: rewrite prompt into variations via text model
# ---------------------------------------------------------------------------
class PromptRewriteWorker(QThread):
    """Call the chat completions API to rewrite a user prompt into N variations."""
    finished = pyqtSignal(list)   # list[str] of rewritten prompts
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, original_prompt: str, count: int):
        super().__init__()
        self.original_prompt = original_prompt
        self.count = count

    def run(self):
        try:
            self.progress.emit(f"Rewriting prompt into {self.count} variations...")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            }
            system_msg = (
                "You are a creative prompt engineer for an image generation AI. "
                "The user will give you an image prompt. Your task is to rewrite it "
                f"into exactly {self.count} different variations. Each variation should "
                "preserve the core meaning/subject but explore different angles: "
                "lighting, composition, style, mood, color palette, camera angle, "
                "artistic medium, level of detail, atmosphere, etc. "
                "Make each variation distinct and interesting. "
                "Return ONLY a JSON array of strings, no explanation. Example:\n"
                '[\"variation 1\", \"variation 2\", ...]'
            )
            body = {
                "model": TEXT_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": self.original_prompt},
                ],
                "temperature": 1.0,
            }
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                json=body,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"].strip()
            # Parse JSON array from the response (handle markdown fences)
            if content.startswith("```"):
                # Strip ```json ... ``` wrapping
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines).strip()

            variations = json.loads(content)
            if not isinstance(variations, list):
                raise ValueError("Model did not return a JSON array.")
            # Ensure we have exactly the right count
            variations = [str(v) for v in variations[:self.count]]
            if len(variations) < self.count:
                # Pad with original if model returned fewer
                while len(variations) < self.count:
                    variations.append(self.original_prompt)

            self.finished.emit(variations)
        except Exception as e:
            self.error.emit(f"Prompt rewrite failed: {e}")


# ---------------------------------------------------------------------------
# Worker: generate images from a list of prompts (one image per prompt)
# ---------------------------------------------------------------------------
class BatchImageGenWorker(QThread):
    """Generate one image per prompt. Emits results incrementally."""
    image_ready = pyqtSignal(str, str)  # (file_path, prompt_used)
    all_finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, prompts: list[str], aspect_ratio: str):
        super().__init__()
        self.prompts = prompts
        self.aspect_ratio = aspect_ratio

    def run(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        total = len(self.prompts)
        for idx, prompt in enumerate(self.prompts):
            try:
                self.progress.emit(f"Generating image {idx+1}/{total}...")
                body = {
                    "model": "grok-imagine-image-pro",
                    "prompt": prompt,
                    "n": 1,
                    "response_format": "url",
                }
                if self.aspect_ratio != "auto":
                    body["aspect_ratio"] = self.aspect_ratio

                resp = requests.post(
                    f"{BASE_URL}/images/generations",
                    json=body,
                    headers=headers,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                images = data.get("data", [])
                if images:
                    url = images[0].get("url", "")
                    if url:
                        ts = int(time.time() * 1000)
                        dest = IMAGES_DIR / f"img_{ts}_{idx}.png"
                        self.progress.emit(f"Downloading image {idx+1}/{total}...")
                        download_file(url, dest)
                        self.image_ready.emit(str(dest), prompt)
                    else:
                        self.progress.emit(f"Image {idx+1}: no URL in response, skipping.")
                else:
                    self.progress.emit(f"Image {idx+1}: empty response, skipping.")
            except Exception as e:
                self.progress.emit(f"Image {idx+1} failed: {e}")

        self.all_finished.emit()


# ---------------------------------------------------------------------------
# Worker: video generation (unchanged)
# ---------------------------------------------------------------------------
class VideoGenWorker(QThread):
    finished = pyqtSignal(str)  # local video path
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, prompt, image_path=None, duration=5, aspect_ratio="16:9", resolution="720p"):
        super().__init__()
        self.prompt = prompt
        self.image_path = image_path
        self.duration = duration
        self.aspect_ratio = aspect_ratio
        self.resolution = resolution

    def run(self):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            }
            body = {
                "model": "grok-imagine-video",
                "prompt": self.prompt,
                "duration": self.duration,
                "aspect_ratio": self.aspect_ratio,
                "resolution": self.resolution,
            }
            if self.image_path:
                img_data = Path(self.image_path).read_bytes()
                b64 = base64.b64encode(img_data).decode()
                ext = Path(self.image_path).suffix.lstrip(".").lower()
                if ext == "jpg":
                    ext = "jpeg"
                # REST API uses nested object: {"image": {"url": "data:..."}}
                body["image"] = {"url": f"data:image/{ext};base64,{b64}"}

            self.progress.emit("Submitting video generation request...")
            resp = requests.post(
                f"{BASE_URL}/videos/generations",
                json=body,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            request_id = data.get("request_id")
            if not request_id:
                self.error.emit("No request_id returned from API.")
                return

            # Poll for completion
            self.progress.emit("Video is being generated (this may take a few minutes)...")
            poll_url = f"{BASE_URL}/videos/{request_id}"
            poll_count = 0
            max_polls = 180  # 15 minutes max (180 * 5s)
            while poll_count < max_polls:
                time.sleep(5)
                poll_count += 1
                poll_resp = requests.get(poll_url, headers=headers, timeout=60)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
                status = poll_data.get("status", "")

                # Try to find video URL in response (handles different response shapes)
                video_url = ""
                if isinstance(poll_data.get("video"), dict):
                    video_url = poll_data["video"].get("url", "")
                elif isinstance(poll_data.get("url"), str):
                    video_url = poll_data["url"]

                if status.lower() in ("done", "completed", "complete", "success") or video_url:
                    if not video_url:
                        self.error.emit(f"Video completed but no URL found. Response: {poll_data}")
                        return
                    ts = int(time.time() * 1000)
                    dest = VIDEOS_DIR / f"video_{ts}.mp4"
                    self.progress.emit("Downloading video...")
                    download_file(video_url, dest)
                    self.finished.emit(str(dest))
                    return
                elif status.lower() in ("expired", "failed", "error"):
                    self.error.emit(f"Video generation failed (status: {status}). Response: {poll_data}")
                    return
                else:
                    elapsed = poll_count * 5
                    mins, secs = divmod(elapsed, 60)
                    time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
                    self.progress.emit(f"Generating video... {time_str} elapsed (poll #{poll_count})")

            self.error.emit("Video generation timed out after 15 minutes.")
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Clickable image thumbnail with prompt label
# ---------------------------------------------------------------------------
class ImageCard(QFrame):
    clicked = pyqtSignal(str)  # emit file path

    def __init__(self, file_path: str, prompt: str = "", parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.prompt = prompt
        self.selected = False
        self.setFixedSize(260, 290)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._setup_ui()
        self._apply_style(False)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        pixmap = QPixmap(self.file_path)
        if pixmap.isNull():
            label = QLabel("Failed to load")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
        else:
            label = QLabel()
            label.setPixmap(pixmap.scaled(
                248, 210,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

        # Prompt snippet
        if self.prompt:
            snippet = self.prompt if len(self.prompt) <= 60 else self.prompt[:57] + "..."
            prompt_label = QLabel(snippet)
            prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            prompt_label.setWordWrap(True)
            prompt_label.setStyleSheet("font-size: 10px; color: #8b949e; padding: 0 2px;")
            prompt_label.setMaximumHeight(36)
            prompt_label.setToolTip(self.prompt)
            layout.addWidget(prompt_label)

        name_label = QLabel(Path(self.file_path).name)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("font-size: 9px; color: #555;")
        layout.addWidget(name_label)

    def _apply_style(self, selected):
        if selected:
            self.setStyleSheet("""
                ImageCard {
                    border: 3px solid #4fc3f7;
                    border-radius: 8px;
                    background: #1e2a38;
                }
            """)
        else:
            self.setStyleSheet("""
                ImageCard {
                    border: 2px solid #333;
                    border-radius: 8px;
                    background: #1a1a2e;
                }
                ImageCard:hover {
                    border: 2px solid #555;
                }
            """)

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self._apply_style(self.selected)
        self.clicked.emit(self.file_path)

    def set_selected(self, sel: bool):
        self.selected = sel
        self._apply_style(sel)


# ---------------------------------------------------------------------------
# Flow layout
# ---------------------------------------------------------------------------
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=6, spacing=8):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._margin = margin
        self._spacing = spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize(0, 0)
        for item in self._items:
            s = s.expandedTo(item.minimumSize())
        m = self._margin
        return s + QSize(2 * m, 2 * m)

    def _do_layout(self, rect, test_only=False):
        x = rect.x() + self._margin
        y = rect.y() + self._margin
        line_height = 0
        right = rect.right() - self._margin

        for item in self._items:
            w = item.sizeHint().width()
            h = item.sizeHint().height()
            if x + w > right and line_height > 0:
                x = rect.x() + self._margin
                y += line_height + self._spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x += w + self._spacing
            line_height = max(line_height, h)

        return y + line_height - rect.y() + self._margin


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class GrokStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grok Studio")
        self.setMinimumSize(1100, 760)
        self._selected_image: str | None = None
        self._selected_prompt: str = ""
        self._browsed_image: str | None = None  # image from file browser
        self._all_image_paths: list[str] = []
        self._image_cards: list[ImageCard] = []
        self._current_page = 0
        self._last_prompt = ""
        self._last_variations: list[str] = []

        self._build_ui()
        self._apply_dark_theme()

    # ---------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left panel: image generation ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title = QLabel("Grok Studio")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #4fc3f7; margin: 4px 0;")
        left_layout.addWidget(title)

        # Prompt
        prompt_group = QGroupBox("Image Prompt")
        pg_layout = QVBoxLayout(prompt_group)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Describe the image you want to generate...\n"
            "The text model will rewrite your prompt into multiple creative variations."
        )
        self.prompt_edit.setMaximumHeight(100)
        pg_layout.addWidget(self.prompt_edit)

        # Options row
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Variations:"))
        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(1, 10)
        self.variations_spin.setValue(4)
        self.variations_spin.setToolTip("Number of prompt variations to generate (each produces 1 image)")
        opts.addWidget(self.variations_spin)

        opts.addWidget(QLabel("Aspect:"))
        self.img_aspect_combo = QComboBox()
        self.img_aspect_combo.addItems([
            "auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3",
        ])
        opts.addWidget(self.img_aspect_combo)
        opts.addStretch()
        pg_layout.addLayout(opts)

        # Generate & Next buttons
        btn_row = QHBoxLayout()
        self.gen_img_btn = QPushButton("Generate Images")
        self.gen_img_btn.setMinimumHeight(36)
        self.gen_img_btn.setToolTip("Rewrite prompt into variations, then generate one image per variation")
        self.gen_img_btn.clicked.connect(self._on_generate_images)
        btn_row.addWidget(self.gen_img_btn)

        self.next_page_btn = QPushButton("Next Page (Re-generate)")
        self.next_page_btn.setMinimumHeight(36)
        self.next_page_btn.setEnabled(False)
        self.next_page_btn.setToolTip("Generate new variations and images with the same base prompt")
        self.next_page_btn.clicked.connect(self._on_next_page)
        btn_row.addWidget(self.next_page_btn)
        pg_layout.addLayout(btn_row)

        left_layout.addWidget(prompt_group)

        # Image gallery
        gallery_group = QGroupBox("Generated Images (click to select)")
        gallery_layout = QVBoxLayout(gallery_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.gallery_widget = QWidget()
        self.gallery_flow = FlowLayout(self.gallery_widget)
        scroll.setWidget(self.gallery_widget)
        gallery_layout.addWidget(scroll)

        # Gallery actions
        gal_acts = QHBoxLayout()
        self.open_img_folder_btn = QPushButton("Open Images Folder")
        self.open_img_folder_btn.clicked.connect(lambda: self._open_in_explorer(IMAGES_DIR))
        gal_acts.addWidget(self.open_img_folder_btn)

        self.open_selected_btn = QPushButton("Open Selected Image")
        self.open_selected_btn.setEnabled(False)
        self.open_selected_btn.clicked.connect(self._open_selected_image)
        gal_acts.addWidget(self.open_selected_btn)
        gallery_layout.addLayout(gal_acts)

        left_layout.addWidget(gallery_group, stretch=1)

        # Status
        self.img_status = QLabel("")
        self.img_status.setStyleSheet("color: #aaa; font-size: 12px;")
        self.img_status.setWordWrap(True)
        left_layout.addWidget(self.img_status)

        self.img_progress = QProgressBar()
        self.img_progress.setRange(0, 0)
        self.img_progress.setVisible(False)
        left_layout.addWidget(self.img_progress)

        splitter.addWidget(left)

        # ---- Right panel: video generation & playback ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Video generation
        vid_group = QGroupBox("Video Generation")
        vg_layout = QVBoxLayout(vid_group)

        # Image source section
        img_src_label = QLabel("Source Image:")
        img_src_label.setStyleSheet("font-weight: bold; color: #8b949e;")
        vg_layout.addWidget(img_src_label)

        self.selected_img_label = QLabel("No image selected.")
        self.selected_img_label.setStyleSheet("color: #888;")
        self.selected_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_img_label.setWordWrap(True)
        vg_layout.addWidget(self.selected_img_label)

        self.selected_img_preview = QLabel()
        self.selected_img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_img_preview.setFixedHeight(150)
        self.selected_img_preview.setVisible(False)
        vg_layout.addWidget(self.selected_img_preview)

        # Image source buttons
        img_src_btns = QHBoxLayout()
        self.browse_img_btn = QPushButton("Browse Local Image...")
        self.browse_img_btn.setToolTip("Choose an image file from your computer")
        self.browse_img_btn.clicked.connect(self._on_browse_image)
        img_src_btns.addWidget(self.browse_img_btn)

        self.clear_img_btn = QPushButton("Clear Image")
        self.clear_img_btn.setToolTip("Remove the selected image (text-to-video mode)")
        self.clear_img_btn.clicked.connect(self._on_clear_video_image)
        img_src_btns.addWidget(self.clear_img_btn)
        vg_layout.addLayout(img_src_btns)

        # Use gallery image checkbox
        self.use_gallery_check = QCheckBox("Use selected gallery image (overrides browse)")
        self.use_gallery_check.setChecked(True)
        self.use_gallery_check.setToolTip("When checked, the image selected in the gallery is used instead of a browsed file")
        vg_layout.addWidget(self.use_gallery_check)

        vg_layout.addWidget(QLabel("Video Prompt (describe the motion/animation):"))
        self.video_prompt_edit = QTextEdit()
        self.video_prompt_edit.setPlaceholderText(
            "Describe the desired motion and animation, e.g.:\n"
            "\"Slow camera pan to the right, clouds drifting, gentle wind in the hair\""
        )
        self.video_prompt_edit.setMaximumHeight(70)
        vg_layout.addWidget(self.video_prompt_edit)

        # Video options
        vopts = QHBoxLayout()
        vopts.addWidget(QLabel("Duration(s):"))
        self.vid_duration_spin = QSpinBox()
        self.vid_duration_spin.setRange(1, 15)
        self.vid_duration_spin.setValue(5)
        vopts.addWidget(self.vid_duration_spin)

        vopts.addWidget(QLabel("Aspect:"))
        self.vid_aspect_combo = QComboBox()
        self.vid_aspect_combo.addItems(["16:9", "1:1", "9:16", "4:3", "3:4"])
        vopts.addWidget(self.vid_aspect_combo)

        vopts.addWidget(QLabel("Resolution:"))
        self.vid_res_combo = QComboBox()
        self.vid_res_combo.addItems(["720p", "480p"])
        vopts.addWidget(self.vid_res_combo)
        vopts.addStretch()
        vg_layout.addLayout(vopts)

        self.gen_vid_btn = QPushButton("Generate Video")
        self.gen_vid_btn.setMinimumHeight(36)
        self.gen_vid_btn.clicked.connect(self._on_generate_video)
        vg_layout.addWidget(self.gen_vid_btn)

        right_layout.addWidget(vid_group)

        # Video player
        player_group = QGroupBox("Video Player")
        pl_layout = QVBoxLayout(player_group)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(280)
        pl_layout.addWidget(self.video_widget)

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

        # Controls
        ctrl = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(70)
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        ctrl.addWidget(self.play_btn)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setFixedWidth(90)
        ctrl.addWidget(self.time_label)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderMoved.connect(self._seek)
        ctrl.addWidget(self.seek_slider)
        pl_layout.addLayout(ctrl)

        # Video file actions
        vid_acts = QHBoxLayout()
        self.open_vid_folder_btn = QPushButton("Open Videos Folder")
        self.open_vid_folder_btn.clicked.connect(lambda: self._open_in_explorer(VIDEOS_DIR))
        vid_acts.addWidget(self.open_vid_folder_btn)

        self.open_vid_file_btn = QPushButton("Open Video File Location")
        self.open_vid_file_btn.setEnabled(False)
        self.open_vid_file_btn.clicked.connect(self._open_current_video_location)
        vid_acts.addWidget(self.open_vid_file_btn)
        pl_layout.addLayout(vid_acts)

        right_layout.addWidget(player_group, stretch=1)

        # Video status
        self.vid_status = QLabel("")
        self.vid_status.setStyleSheet("color: #aaa; font-size: 12px;")
        right_layout.addWidget(self.vid_status)

        self.vid_progress = QProgressBar()
        self.vid_progress.setRange(0, 0)
        self.vid_progress.setVisible(False)
        right_layout.addWidget(self.vid_progress)

        splitter.addWidget(right)
        splitter.setSizes([540, 460])

        # Timer for media position updates
        self._pos_timer = QTimer(self)
        self._pos_timer.setInterval(250)
        self._pos_timer.timeout.connect(self._update_position)

        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackStateChanged.connect(self._playback_state_changed)

        self._current_video_path: str | None = None

    # ---------------------------------------------------------------
    # Dark theme
    # ---------------------------------------------------------------
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #e6edf3;
                font-family: "Segoe UI", sans-serif;
            }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 14px;
                font-weight: bold;
                color: #8b949e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTextEdit, QLineEdit {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 4px;
                color: #e6edf3;
                padding: 6px;
                font-size: 13px;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 1px solid #4fc3f7;
            }
            QPushButton {
                background: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #e6edf3;
                padding: 6px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #30363d;
                border-color: #4fc3f7;
            }
            QPushButton:pressed {
                background: #0d419d;
            }
            QPushButton:disabled {
                color: #484f58;
                background: #161b22;
                border-color: #21262d;
            }
            QComboBox {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 4px;
                color: #e6edf3;
                padding: 4px 8px;
            }
            QComboBox QAbstractItemView {
                background: #161b22;
                color: #e6edf3;
                selection-background-color: #0d419d;
            }
            QSpinBox {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 4px;
                color: #e6edf3;
                padding: 4px;
            }
            QScrollArea {
                border: none;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #30363d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4fc3f7;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #1a6fb5;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #30363d;
                border-radius: 4px;
                text-align: center;
                color: #e6edf3;
                background: #161b22;
                height: 8px;
            }
            QProgressBar::chunk {
                background: #4fc3f7;
                border-radius: 3px;
            }
            QCheckBox {
                color: #8b949e;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #30363d;
                border-radius: 3px;
                background: #161b22;
            }
            QCheckBox::indicator:checked {
                background: #4fc3f7;
                border-color: #4fc3f7;
            }
            QLabel {
                color: #e6edf3;
            }
            QSplitter::handle {
                background: #30363d;
                width: 2px;
            }
        """)

    # ---------------------------------------------------------------
    # Image generation: rewrite prompt -> batch generate
    # ---------------------------------------------------------------
    def _on_generate_images(self):
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing Prompt", "Please enter a prompt for image generation.")
            return
        if not API_KEY:
            QMessageBox.critical(self, "No API Key", "Set XAI_API_KEY environment variable.")
            return

        self._last_prompt = prompt
        self._current_page = 0
        self._all_image_paths.clear()
        self._clear_gallery()
        self._set_img_busy(True)

        count = self.variations_spin.value()
        self._start_rewrite(prompt, count)

    def _on_next_page(self):
        if not self._last_prompt:
            return
        self._current_page += 1
        self._set_img_busy(True)
        count = self.variations_spin.value()
        self._start_rewrite(self._last_prompt, count)

    def _start_rewrite(self, prompt: str, count: int):
        """Step 1: rewrite the prompt into variations."""
        self._rewrite_worker = PromptRewriteWorker(prompt, count)
        self._rewrite_worker.finished.connect(self._on_rewrite_done)
        self._rewrite_worker.error.connect(self._on_img_error)
        self._rewrite_worker.progress.connect(lambda msg: self.img_status.setText(msg))
        self._rewrite_worker.start()

    def _on_rewrite_done(self, variations: list[str]):
        """Step 2: got variations, now generate images."""
        self._last_variations = variations
        self.img_status.setText(
            f"Got {len(variations)} prompt variations. Starting image generation..."
        )
        aspect = self.img_aspect_combo.currentText()

        self._batch_worker = BatchImageGenWorker(variations, aspect)
        self._batch_worker.image_ready.connect(self._on_single_image_ready)
        self._batch_worker.all_finished.connect(self._on_all_images_done)
        self._batch_worker.error.connect(self._on_img_error)
        self._batch_worker.progress.connect(lambda msg: self.img_status.setText(msg))
        self._batch_worker.start()

    def _on_single_image_ready(self, path: str, prompt: str):
        """Called each time one image is downloaded — add to gallery immediately."""
        self._all_image_paths.append(path)
        card = ImageCard(path, prompt)
        card.clicked.connect(self._on_image_selected)
        self.gallery_flow.addWidget(card)
        self._image_cards.append(card)

    def _on_all_images_done(self):
        self._set_img_busy(False)
        total = len(self._all_image_paths)
        self.img_status.setText(f"Done! {total} image(s) generated.")
        self.next_page_btn.setEnabled(True)

    def _on_img_error(self, msg: str):
        self._set_img_busy(False)
        self.img_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Image Generation Error", msg)

    def _set_img_busy(self, busy: bool):
        self.gen_img_btn.setEnabled(not busy)
        self.next_page_btn.setEnabled(not busy and len(self._all_image_paths) > 0)
        self.img_progress.setVisible(busy)
        if busy:
            self.img_status.setText("Generating...")

    def _clear_gallery(self):
        for card in self._image_cards:
            card.setParent(None)
            card.deleteLater()
        self._image_cards.clear()

    def _on_image_selected(self, path: str):
        # Deselect all others
        for card in self._image_cards:
            if card.file_path != path:
                card.set_selected(False)

        # Toggle
        sender_card = next((c for c in self._image_cards if c.file_path == path), None)
        if sender_card and sender_card.selected:
            self._selected_image = path
            self._selected_prompt = sender_card.prompt
            self.open_selected_btn.setEnabled(True)
        else:
            self._selected_image = None
            self._selected_prompt = ""
            self.open_selected_btn.setEnabled(False)

        # Update the video panel preview
        self._update_video_image_preview()

    # ---------------------------------------------------------------
    # Video: image source management
    # ---------------------------------------------------------------
    def _on_browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image for Video",
            str(Path(__file__).parent),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*)",
        )
        if path:
            self._browsed_image = path
            self._update_video_image_preview()

    def _on_clear_video_image(self):
        self._browsed_image = None
        self._update_video_image_preview()

    def _get_video_image_path(self) -> str | None:
        """Determine which image to use for video generation."""
        # Gallery image takes priority when checkbox is checked and an image is selected
        if self.use_gallery_check.isChecked() and self._selected_image:
            return self._selected_image
        # Otherwise use browsed image
        if self._browsed_image:
            return self._browsed_image
        # Fallback: gallery image even if checkbox unchecked
        return self._selected_image

    def _update_video_image_preview(self):
        """Update the preview and label based on current image source."""
        path = self._get_video_image_path()
        if path and Path(path).exists():
            self.selected_img_label.setText(f"Image: {Path(path).name}")
            pixmap = QPixmap(path).scaled(
                200, 140, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.selected_img_preview.setPixmap(pixmap)
            self.selected_img_preview.setVisible(True)
        else:
            self.selected_img_label.setText("No image selected (text-to-video mode).")
            self.selected_img_preview.setVisible(False)

    # ---------------------------------------------------------------
    # Video generation
    # ---------------------------------------------------------------
    def _on_generate_video(self):
        video_prompt = self.video_prompt_edit.toPlainText().strip()
        if not video_prompt:
            # For image-to-video, a motion prompt is still required
            video_prompt = "Animate this image with gentle natural motion"

        # For image-to-video: prompt should describe motion, not the image content
        # Only add the original image prompt if there's no source image
        image_path = self._get_video_image_path()
        if image_path:
            # Image-to-video: use only the motion prompt
            prompt = video_prompt
        else:
            # Text-to-video: combine image description + motion
            prompt_parts = []
            if self._selected_prompt:
                prompt_parts.append(self._selected_prompt)
            elif self._last_prompt:
                prompt_parts.append(self._last_prompt)
            if video_prompt:
                prompt_parts.append(video_prompt)
            prompt = ". ".join(prompt_parts) if prompt_parts else "Generate a video"

        if not API_KEY:
            QMessageBox.critical(self, "No API Key", "Set XAI_API_KEY environment variable.")
            return

        duration = self.vid_duration_spin.value()
        aspect = self.vid_aspect_combo.currentText()
        resolution = self.vid_res_combo.currentText()

        self._set_vid_busy(True)

        self._vid_worker = VideoGenWorker(prompt, image_path, duration, aspect, resolution)
        self._vid_worker.finished.connect(self._on_video_ready)
        self._vid_worker.error.connect(self._on_vid_error)
        self._vid_worker.progress.connect(lambda msg: self.vid_status.setText(msg))
        self._vid_worker.start()

    def _on_video_ready(self, path: str):
        self._set_vid_busy(False)
        self._current_video_path = path
        self.vid_status.setText(f"Video ready: {Path(path).name}")
        self.open_vid_file_btn.setEnabled(True)
        self._load_video(path)

    def _on_vid_error(self, msg: str):
        self._set_vid_busy(False)
        self.vid_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Video Generation Error", msg)

    def _set_vid_busy(self, busy: bool):
        self.gen_vid_btn.setEnabled(not busy)
        self.vid_progress.setVisible(busy)
        if busy:
            self.vid_status.setText("Generating video...")

    # ---------------------------------------------------------------
    # Video playback
    # ---------------------------------------------------------------
    def _load_video(self, path: str):
        self.player.setSource(QUrl.fromLocalFile(path))
        self.play_btn.setEnabled(True)
        self.play_btn.setText("Play")
        self.player.play()

    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setText("Pause")
            self._pos_timer.start()
        else:
            self.play_btn.setText("Play")
            self._pos_timer.stop()

    def _duration_changed(self, duration):
        self.seek_slider.setRange(0, duration)

    def _update_position(self):
        pos = self.player.position()
        dur = self.player.duration()
        self.seek_slider.setValue(pos)
        self.time_label.setText(f"{self._fmt(pos)} / {self._fmt(dur)}")

    def _seek(self, pos):
        self.player.setPosition(pos)

    @staticmethod
    def _fmt(ms):
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"

    # ---------------------------------------------------------------
    # Explorer helpers
    # ---------------------------------------------------------------
    def _open_in_explorer(self, path: Path):
        os.startfile(str(path))

    def _open_selected_image(self):
        if self._selected_image:
            subprocess.Popen(f'explorer /select,"{self._selected_image}"')

    def _open_current_video_location(self):
        if self._current_video_path:
            subprocess.Popen(f'explorer /select,"{self._current_video_path}"')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = GrokStudioWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
