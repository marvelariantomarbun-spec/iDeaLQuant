import sys
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys
import time

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtWidgets import QApplication
from src.ui.widgets.data_panel import DataPanel
from src.ui.widgets.optimizer_panel import OptimizerPanel
from src.ui.widgets.validation_panel import ValidationPanel
from src.ui.widgets.strategy_panel import StrategyPanel
from src.ui.widgets.export_panel import ExportPanel
from src.core.database import db

def main():
    app = QApplication(sys.argv)
    
    print("Creating Panels...")
    data_panel = DataPanel()
    strategy_panel = StrategyPanel()
    optimizer_panel = OptimizerPanel()
    validation_panel = ValidationPanel()
    export_panel = ExportPanel()
    
    print("Connecting Signals...")
    # Simulate MainWindow connections
    data_panel.data_loaded.connect(strategy_panel.set_data)
    data_panel.data_loaded.connect(optimizer_panel.set_data)
    data_panel.data_loaded.connect(validation_panel.set_data)
    
    data_panel.process_created.connect(optimizer_panel.set_process)
    # ValidationPanel set_process is NOT connected in MainWindow, so removing it from here to match reality
    # data_panel.process_created.connect(validation_panel.set_process) 
    
    # Fix ExportPanel connection using lambda as in MainWindow
    validation_panel.validation_complete.connect(lambda pid, params: export_panel._refresh_processes())

    
    print("Simulating Auto-Load...")
    # We can trigger it manually if the timer misses or we want to be explicit
    # But let's let the timer run its course by processing events
    
    # Force auto-load logic to run now
    data_panel._try_auto_load_last_session()
    
    print("Processing Events... (Waiting for crash)")
    
    start = time.time()
    while time.time() - start < 10:
        app.processEvents()
        time.sleep(0.1)
        
    print("No crash occurred in 10 seconds.")
    sys.exit(0)

if __name__ == "__main__":
    main()
