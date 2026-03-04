import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

class OverlayWindow(QWidget):
    # Signal to update text from another thread
    text_updated = pyqtSignal(str, str) # (text, type: 'user' | 'tran' | 'partial')

    def __init__(self):
        super().__init__()
        self.history = [] # List to store last 4 translations
        self.max_history = 4
        self.init_ui()
        
        self.clear_timer = QTimer()
        self.clear_timer.timeout.connect(self.clear_partial_only)

    def init_ui(self):
        # 1. ToolTip flag is very aggressive for staying on top on macOS/Windows
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.ToolTip | # Helps keep it above full-screen apps and active windows
            Qt.WindowType.WindowTransparentForInput # Click-through
        )
        
        # 2. Translucent background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # 3. Ensure it's always on top of the stack
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)
        # 4. Don't take focus
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # We will have up to 4 labels for history
        self.labels = []
        for i in range(self.max_history):
            label = QLabel("")
            label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
            # Older translations are more transparent
            opacity = int(255 * (0.3 + (i / self.max_history) * 0.7))
            label.setStyleSheet(f"color: rgba(0, 255, 0, {opacity}); background-color: rgba(0, 0, 0, 160); padding: 12px; border-radius: 8px;")
            label.setWordWrap(True)
            label.setFixedWidth(770)
            # Important: let the label height grow
            label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            label.hide()
            self.layout.addWidget(label)
            self.labels.append(label)

        # Special label for partial results (current speaking)
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

        # Position: TOP middle of the screen
        screen = QApplication.primaryScreen().geometry()
        self.fixed_width = 800
        
        # Initial geometry - height will be managed by adjustSize()
        self.setGeometry(
            (screen.width() - self.fixed_width) // 2,
            40, # 40px from top
            self.fixed_width,
            10 # Start small
        )

        self.text_updated.connect(self.handle_signal)

    def handle_signal(self, text, msg_type):
        if msg_type == 'partial':
            self.update_partial(text)
        elif msg_type == 'tran':
            self.add_to_history(text)

    def update_partial(self, text):
        if not text or text.strip() == "":
            self.partial_label.hide()
        else:
            self.partial_label.setText(f"... {text}")
            self.partial_label.show()
            self.clear_timer.start(5000)
        self.adjustSize()

    def add_to_history(self, text):
        self.partial_label.hide()
        self.history.append(text)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Update labels (bottom-most label in layout is the newest)
        for i in range(self.max_history):
            # labels[0] is oldest, labels[3] is newest
            # history[0] is oldest, history[-1] is newest
            h_idx = len(self.history) - self.max_history + i
            if h_idx >= 0:
                self.labels[i].setText(self.history[h_idx])
                self.labels[i].show()
            else:
                self.labels[i].hide()
        
        # Force window to resize and recalculate layout
        self.adjustSize()

    def clear_partial_only(self):
        self.partial_label.hide()
        self.clear_timer.stop()
        self.adjustSize()

def run_overlay_app():
    app = QApplication(sys.argv)
    window = OverlayWindow()
    window.show()
    return app, window
