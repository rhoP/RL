import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import os
import json
import logging
from datetime import datetime
import tomlkit


class ExperimentLogger:
    """Class to handle experiment logging, configuration, and result saving."""
    
    def __init__(self, experiment_name: str, base_log_dir: str = "experiments"):
        """
        Initialize the experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            base_log_dir: Base directory for all experiments
        """
        self.experiment_name = experiment_name
        self.base_log_dir = base_log_dir
        self.experiment_dir = self._create_experiment_dir()
        self.config_path = os.path.join(self.experiment_dir, "config.toml")
        self.log_path = os.path.join(self.experiment_dir, "experiment.log")
        
        # Setup logging
        self._setup_logging()
        
    def _create_experiment_dir(self) -> str:
        """Create a unique directory for this experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.base_log_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        # Create logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def save_config(self, config: Dict):
        """Save configuration to TOML file."""
        # Create TOML document with comments
        doc = tomlkit.document()
        doc.add(tomlkit.comment("Experiment Configuration"))
        doc.add(tomlkit.comment(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
        doc.add(tomlkit.comment(""))
        
        # Add configuration sections
        for section_name, section_data in config.items():
            doc.add(tomlkit.comment(f"{section_name.upper()} SECTION"))
            section_table = tomlkit.table()
            for key, value in section_data.items():
                # Convert numpy types to Python native types
                if isinstance(value, (np.int32, np.int64)): # pyright: ignore[reportArgumentType]
                    value = int(value)
                elif isinstance(value, (np.float32, np.float64)): # pyright: ignore[reportArgumentType]
                    value = float(value)
                elif isinstance(value, np.ndarray):
                    value = tomlkit.array(value.tolist())
                section_table.add(key, value)
            doc[section_name] = section_table
            doc.add(tomlkit.comment(""))
        
        # Write to file
        with open(self.config_path, 'w') as f:
            f.write(tomlkit.dumps(doc))
        
        self.logger.info(f"Configuration saved to {self.config_path}")
    
    def save_plot(self, fig, plot_name: str):
        """Save a plot to the experiment directory."""
        plot_path = os.path.join(self.experiment_dir, f"{plot_name}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Plot saved to {plot_path}")
    
    def save_model(self, model, model_name: str):
        """Save a model to the experiment directory."""
        model_path = os.path.join(self.experiment_dir, f"{model_name}.zip")
        model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def save_trajectories(self, trajectories, filename: str = "trajectories.json"):
        """Save trajectories to JSON file."""
        trajectories_path = os.path.join(self.experiment_dir, filename)
        with open(trajectories_path, 'w') as f:
            json.dump(trajectories, f, indent=2)
        self.logger.info(f"Trajectories saved to {trajectories_path}")
    
    def log_metrics(self, metrics: Dict):
        """Log metrics to file and console."""
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")