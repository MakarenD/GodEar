import sys
from pynput import keyboard
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QSizePolicy, QSystemTrayIcon, QMenu, QDialog, 
                             QComboBox, QPushButton, QHBoxLayout, QCheckBox, QStyle,
                             QLineEdit, QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QFont, QIcon, QAction, QKeySequence
from tts_engine import TTSEngine

class HotkeyLineEdit(QLineEdit):
    def __init__(self, default_keybind, parent=None):
        super().__init__(default_keybind, parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Click to set...")

    def mousePressEvent(self, event):
        self.setText("Press keys...")
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta, Qt.Key.Key_Super_L, Qt.Key.Key_Super_R, Qt.Key.Key_AltGr):
            return

        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append('<ctrl>')
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append('<alt>')
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append('<shift>')
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            parts.append('<cmd>')

        special_keys = {
            Qt.Key.Key_Escape: '<esc>', Qt.Key.Key_Space: '<space>',
            Qt.Key.Key_Tab: '<tab>', Qt.Key.Key_Enter: '<enter>',
            Qt.Key.Key_Return: '<enter>', Qt.Key.Key_Backspace: '<backspace>',
            Qt.Key.Key_Delete: '<delete>', Qt.Key.Key_Up: '<up>',
            Qt.Key.Key_Down: '<down>', Qt.Key.Key_Left: '<left>',
            Qt.Key.Key_Right: '<right>', Qt.Key.Key_Insert: '<insert>',
            Qt.Key.Key_Home: '<home>', Qt.Key.Key_End: '<end>',
            Qt.Key.Key_PageUp: '<page_up>', Qt.Key.Key_PageDown: '<page_down>'
        }

        if key in special_keys:
            parts.append(special_keys[key])
        elif Qt.Key.Key_F1 <= key <= Qt.Key.Key_F35:
            parts.append(f"f{key - Qt.Key.Key_F1 + 1}")
        else:
            key_str = QKeySequence(key).toString().lower()
            if key_str:
                parts.append(key_str)

        if parts:
            self.setText('+'.join(parts))
            self.clearFocus()

class HotkeyManager(QObject):
    toggle_visibility_sig = pyqtSignal()
    toggle_drag_mode_sig = pyqtSignal()
    toggle_mute_sig = pyqtSignal()

    def __init__(self, keybinds=None):
        super().__init__()
        self.active_hotkeys = []
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.update_keybinds(keybinds or {
            'visibility': '<ctrl>+<alt>+h',
            'drag': '<ctrl>+<alt>+d',
            'mute': '<ctrl>+<alt>+m'
        })

    def update_keybinds(self, keybinds):
        mapping = {}
        if keybinds.get('visibility'): mapping[keybinds['visibility']] = self.on_toggle_visibility
        if keybinds.get('drag'): mapping[keybinds['drag']] = self.on_toggle_drag_mode
        if keybinds.get('mute'): mapping[keybinds['mute']] = self.on_toggle_mute
            
        new_hotkeys = []
        for k, v in mapping.items():
            try:
                new_hotkeys.append(keyboard.HotKey(keyboard.HotKey.parse(k), v))
            except Exception as e:
                print(f"Failed to parse hotkey {k}: {e}")
        self.active_hotkeys = new_hotkeys

    def on_press(self, key):
        key = self.listener.canonical(key)
        for hk in self.active_hotkeys:
            try:
                hk.press(key)
            except Exception:
                pass

    def on_release(self, key):
        key = self.listener.canonical(key)
        for hk in self.active_hotkeys:
            try:
                hk.release(key)
            except Exception:
                pass

    def on_toggle_visibility(self):
        self.toggle_visibility_sig.emit()

    def on_toggle_drag_mode(self):
        self.toggle_drag_mode_sig.emit()

    def on_toggle_mute(self):
        self.toggle_mute_sig.emit()

class SettingsWindow(QDialog):
    def __init__(self, current_settings, audio_devices, parent=None):
        super().__init__(parent)
        self.settings = current_settings
        self.audio_devices = audio_devices # List of (id, name)
        self.output_devices = TTSEngine.get_output_devices()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Settings - Speech Translation")
        self.setFixedWidth(400)
        layout = QVBoxLayout()

        # Languages
        lang_layout = QHBoxLayout()
        self.from_lang_cb = QComboBox()
        # Add some common languages, but ideally this should be a larger list
        langs = ["en", "ru", "de", "fr", "es", "it", "zh", "tr"]
        self.from_lang_cb.addItems(langs)
        self.from_lang_cb.setCurrentText(self.settings.get('from_lang', 'en'))
        
        self.to_lang_cb = QComboBox()
        self.to_lang_cb.addItems(langs)
        self.to_lang_cb.setCurrentText(self.settings.get('to_lang', 'ru'))
        self.to_lang_cb.currentTextChanged.connect(self.update_voice_list)
        
        lang_layout.addWidget(QLabel("From:"))
        lang_layout.addWidget(self.from_lang_cb)
        lang_layout.addWidget(QLabel("To:"))
        lang_layout.addWidget(self.to_lang_cb)
        layout.addLayout(lang_layout)

        # Engine
        self.engine_cb = QComboBox()
        self.engine_cb.addItems(["vosk", "whisper", "whisper-lite", "vosk+whisper", "vosk+whisper-lite"])
        self.engine_cb.setCurrentText(self.settings.get('engine', 'vosk'))
        layout.addWidget(QLabel("STT Engine:"))
        layout.addWidget(self.engine_cb)

        # Translation Type
        self.translation_type_cb = QComboBox()
        self.translation_type_cb.addItems(["online", "offline"])
        self.translation_type_cb.setCurrentText(self.settings.get('translation_type', 'online'))
        layout.addWidget(QLabel("Translation Type:"))
        layout.addWidget(self.translation_type_cb)

        # Audio Device
        self.device_cb = QComboBox()
        self.device_cb.addItem("Default Microphone", None)
        for idx, name in self.audio_devices:
            self.device_cb.addItem(name, idx)
        
        # Select current device if possible
        current_dev = self.settings.get('device_id')
        if current_dev is not None:
            index = self.device_cb.findData(current_dev)
            if index >= 0: self.device_cb.setCurrentIndex(index)

        layout.addWidget(QLabel("Audio Input:"))
        layout.addWidget(self.device_cb)

        # Loopback checkbox
        self.loopback_ch = QCheckBox("Capture System Audio (Loopback)")
        self.loopback_ch.setChecked(self.settings.get('loopback', False))
        layout.addWidget(self.loopback_ch)
        
        # Diarization checkbox
        self.diarization_ch = QCheckBox("Enable Speaker Diarization")
        self.diarization_ch.setChecked(self.settings.get('diarization', False))
        layout.addWidget(self.diarization_ch)

        # TTS Settings
        tts_group = QGroupBox("Voice Output (TTS)")
        tts_layout = QFormLayout()
        
        self.tts_enabled_ch = QCheckBox("Enable Speech Synthesis")
        self.tts_enabled_ch.setChecked(self.settings.get('tts_enabled', False))
        
        self.tts_device_cb = QComboBox()
        self.tts_device_cb.addItem("Default Output", None)
        for idx, name in self.output_devices:
            self.tts_device_cb.addItem(name, idx)
        
        current_tts_dev = self.settings.get('tts_output_device')
        if current_tts_dev is not None:
            idx = self.tts_device_cb.findData(current_tts_dev)
            if idx >= 0: self.tts_device_cb.setCurrentIndex(idx)
            
        self.tts_voice_cb = QComboBox()
        self.update_voice_list(self.to_lang_cb.currentText())
        current_voice = self.settings.get('tts_voice', 'aidar')
        self.tts_voice_cb.setCurrentText(current_voice)
        
        tts_layout.addRow(self.tts_enabled_ch)
        tts_layout.addRow("Output Device:", self.tts_device_cb)
        tts_layout.addRow("Voice:", self.tts_voice_cb)
        tts_group.setLayout(tts_layout)
        layout.addWidget(tts_group)

        # Keybinds
        keybinds_group = QGroupBox("Keybinds (Click to change)")
        keybinds_layout = QFormLayout()
        
        current_kb = self.settings.get('keybinds', {})
        
        self.kb_vis_input = HotkeyLineEdit(current_kb.get('visibility', '<ctrl>+<alt>+h'))
        self.kb_vis_input.setToolTip("Show or hide the translation overlay on the screen.")
        
        self.kb_drag_input = HotkeyLineEdit(current_kb.get('drag', '<ctrl>+<alt>+d'))
        self.kb_drag_input.setToolTip("Enable/disable moving the overlay with your mouse.")
        
        self.kb_mute_input = HotkeyLineEdit(current_kb.get('mute', '<ctrl>+<alt>+m'))
        self.kb_mute_input.setToolTip("Pause/resume listening to your microphone or system audio.")
        
        keybinds_layout.addRow("Toggle Overlay (Show/Hide):", self.kb_vis_input)
        keybinds_layout.addRow("Toggle Drag Mode (Move):", self.kb_drag_input)
        keybinds_layout.addRow("Toggle Mute (Pause STT):", self.kb_mute_input)
        keybinds_group.setLayout(keybinds_layout)
        layout.addWidget(keybinds_group)

        # Buttons
        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply & Restart")
        apply_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def update_voice_list(self, lang):
        self.tts_voice_cb.clear()
        voices = TTSEngine.get_speakers_for_lang(lang[:2])
        self.tts_voice_cb.addItems(voices)

    def get_new_settings(self):
        return {
            'from_lang': self.from_lang_cb.currentText(),
            'to_lang': self.to_lang_cb.currentText(),
            'engine': self.engine_cb.currentText(),
            'translation_type': self.translation_type_cb.currentText(),
            'device_id': self.device_cb.currentData(),
            'loopback': self.loopback_ch.isChecked(),
            'diarization': self.diarization_ch.isChecked(),
            'tts_enabled': self.tts_enabled_ch.isChecked(),
            'tts_output_device': self.tts_device_cb.currentData(),
            'tts_voice': self.tts_voice_cb.currentText(),
            'keybinds': {
                'visibility': self.kb_vis_input.text().strip(),
                'drag': self.kb_drag_input.text().strip(),
                'mute': self.kb_mute_input.text().strip()
            }
        }

class OverlayWindow(QWidget):
    text_updated = pyqtSignal(str, str, float)
    speaker_updated = pyqtSignal(float, str, str)
    settings_changed = pyqtSignal(dict) # Signal to notify translator thread
    mute_toggled = pyqtSignal(bool)

    def __init__(self, current_settings=None, audio_devices=None):
        super().__init__()
        self.history = []
        self.pending_speakers = {} # id -> (label, color)
        self.max_history = 4
        self.current_settings = current_settings or {}
        self.audio_devices = audio_devices or []
        self.drag_mode = False
        self.is_muted = False
        self.drag_pos = None

        self.init_ui()
        self.init_tray()
        
        self.clear_timer = QTimer()
        self.clear_timer.timeout.connect(self.clear_partial_only)

        self.hotkeys = HotkeyManager(self.current_settings.get('keybinds'))
        self.hotkeys.toggle_visibility_sig.connect(self.toggle_visibility)
        self.hotkeys.toggle_drag_mode_sig.connect(self.toggle_drag_mode)
        self.hotkeys.toggle_mute_sig.connect(self.toggle_mute)

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.ToolTip |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.labels = []
        for i in range(self.max_history):
            label = QLabel("")
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
            label.setStyleSheet("background-color: rgba(0, 0, 0, 160); padding: 12px; border-radius: 8px;")
            label.setWordWrap(True)
            label.setFixedWidth(770)
            label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            label.hide()
            self.layout.addWidget(label)
            self.labels.append(label)

        self.partial_label = QLabel("")
        self.partial_label.setFont(QFont("Arial", 16, QFont.Weight.Normal))
        self.partial_label.setStyleSheet("color: rgba(200, 200, 200, 200); background-color: rgba(0, 0, 0, 120); padding: 8px; border-radius: 6px;")
        self.partial_label.setWordWrap(True)
        self.partial_label.setFixedWidth(770)
        self.partial_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)
        self.partial_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.partial_label.hide()
        self.layout.addWidget(self.partial_label)

        self.setLayout(self.layout)

        screen = QApplication.primaryScreen().geometry()
        self.fixed_width = 800
        self.setGeometry((screen.width() - self.fixed_width) // 2, 40, self.fixed_width, 10)

        self.text_updated.connect(self.handle_signal)
        self.speaker_updated.connect(self.update_speaker)

    def init_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        # Fix: Use QStyle.StandardPixmap for standard icons in PyQt6
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        
        menu = QMenu()
        show_action = QAction("Show/Hide Overlay", self)
        show_action.triggered.connect(self.toggle_visibility)
        
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.instance().quit)
        
        menu.addAction(show_action)
        menu.addAction(settings_action)
        menu.addSeparator()
        menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.show()
        self.tray_icon.setToolTip("Speech Translation Overlay")

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def open_settings(self):
        dialog = SettingsWindow(self.current_settings, self.audio_devices, self)
        if dialog.exec():
            new_settings = dialog.get_new_settings()
            self.current_settings.update(new_settings)
            self.settings_changed.emit(new_settings)
            if 'keybinds' in new_settings:
                try:
                    self.hotkeys.update_keybinds(new_settings['keybinds'])
                except Exception as e:
                    print(f"Error updating hotkeys: {e}")

    def handle_signal(self, text, msg_type, msg_id=0.0):
        # print(f"DEBUG UI: handle_signal text='{text[:20]}...' type={msg_type} id={msg_id}")
        if msg_type == 'partial':
            self.update_partial(text)
        elif msg_type == 'tran':
            self.add_to_history(text, msg_id)
        elif msg_type == 'replace_last':
            self.replace_last_history(text, msg_id)

    def update_partial(self, text):
        if not text or text.strip() == "":
            self.partial_label.hide()
        else:
            self.partial_label.setText(f"... {text}")
            self.partial_label.show()
            self.clear_timer.start(5000)
        self.adjustSize()

    def add_to_history(self, text, msg_id=0.0):
        # print(f"DEBUG UI: add_to_history id={msg_id}")
        self.partial_label.hide()
        
        # Check if speaker was identified before text was ready
        speaker_info = self.pending_speakers.pop(msg_id, (None, None))
        if speaker_info[0] is None:
            # Try fuzzy match for pending speakers
            for pid in list(self.pending_speakers.keys()):
                if abs(pid - msg_id) < 0.5:
                    speaker_info = self.pending_speakers.pop(pid)
                    break

        self.history.append({
            'id': msg_id, 
            'text': text, 
            'speaker': speaker_info[0], 
            'color': speaker_info[1]
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.render_history()

    def replace_last_history(self, text, msg_id=0.0):
        # print(f"DEBUG UI: replace_last_history id={msg_id}")
        self.partial_label.hide()
        
        speaker_info = (None, None)
        if msg_id != 0.0:
            speaker_info = self.pending_speakers.pop(msg_id, (None, None))
            if speaker_info[0] is None:
                for pid in list(self.pending_speakers.keys()):
                    if abs(pid - msg_id) < 0.5:
                        speaker_info = self.pending_speakers.pop(pid)
                        break

        if self.history:
            self.history[-1]['text'] = text
            if msg_id != 0.0:
                self.history[-1]['id'] = msg_id
                if speaker_info[0]:
                    self.history[-1]['speaker'] = speaker_info[0]
                    self.history[-1]['color'] = speaker_info[1]
        else:
            self.history.append({
                'id': msg_id, 
                'text': text, 
                'speaker': speaker_info[0], 
                'color': speaker_info[1]
            })
            if len(self.history) > self.max_history:
                self.history.pop(0)
        self.render_history()

    def update_speaker(self, msg_id, speaker_label, color):
        # print(f"DEBUG UI: update_speaker RECEIVED for msg_id {msg_id} label={speaker_label}")
        found = False
        # Try exact match first, then fuzzy in history
        for item in reversed(self.history):
            if (item['id'] == msg_id or abs(item['id'] - msg_id) < 0.5) and msg_id != 0.0:
                item['speaker'] = speaker_label
                item['color'] = color
                found = True
        
        if found:
            # print(f"DEBUG UI: Speaker {speaker_label} applied to history")
            self.render_history()
        else:
            # Diarization was faster than translation, save for later
            # print(f"DEBUG UI: msg_id {msg_id} not yet in history, saving to pending")
            self.pending_speakers[msg_id] = (speaker_label, color)
            # Cleanup old pending speakers (keep only last 10)
            if len(self.pending_speakers) > 10:
                oldest_key = min(self.pending_speakers.keys())
                self.pending_speakers.pop(oldest_key)

    def render_history(self):
        from PyQt6.QtCore import Qt
        import html
        for i in range(self.max_history):
            h_idx = len(self.history) - self.max_history + i
            if h_idx >= 0:
                item = self.history[h_idx]
                opacity = 0.3 + (i / self.max_history) * 0.7
                text_color = f"rgba(0, 255, 0, {opacity})"
                
                display_text = html.escape(item['text'])
                speaker_html = ""
                if item.get('speaker'):
                    speaker_html = f"<b style='color:{item['color']}'>[{html.escape(item['speaker'])}]</b> "
                
                full_html = f"<html><body><div style='color:{text_color}'>{speaker_html}{display_text}</div></body></html>"
                
                # Use setText to support HTML
                self.labels[i].setText(full_html)
                self.labels[i].show()
            else:
                self.labels[i].hide()
        self.adjustSize()

    def clear_partial_only(self):
        self.partial_label.hide()
        self.clear_timer.stop()
        self.adjustSize()

    def toggle_drag_mode(self):
        self.drag_mode = not self.drag_mode
        self.hide() # Required on macOS to apply flag changes properly
        if self.drag_mode:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowTransparentForInput)
            self.setStyleSheet("background-color: rgba(255, 255, 255, 30); border: 2px dashed red;")
            self.update_partial("Drag Mode ON (Ctrl+Alt+D to disable)")
        else:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowTransparentForInput)
            self.setStyleSheet("")
            self.update_partial("Drag Mode OFF")
        self.show()

    def toggle_mute(self):
        self.is_muted = not self.is_muted
        self.mute_toggled.emit(self.is_muted)
        if self.is_muted:
            self.update_partial("MUTED (Ctrl+Alt+M to unmute)")
        else:
            self.update_partial("UNMUTED")

    def mousePressEvent(self, event):
        if self.drag_mode and event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_mode and event.buttons() & Qt.MouseButton.LeftButton and self.drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            event.accept()

def run_overlay_app(current_settings=None, audio_devices=None):
    app = QApplication(sys.argv)
    # Don't quit if settings window closes
    app.setQuitOnLastWindowClosed(False)
    
    window = OverlayWindow(current_settings, audio_devices)
    window.show()
    return app, window
