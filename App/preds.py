import pandas as pd
import numpy as np
import timesfm
import logging


class Predictions:
    def __init__(self, 
                 data: pd.DataFrame, 
                 target_colm: str,
                 content_length: int = 512, 
                 horizon_length: int = 14, 
                 backend: str = "cpu", 
                 checkpoint: str = "google/timesfm-1.0-200m",
                 ) -> None:
        
        logging.info("Initializing Predictions class")
        
        self.data = data.copy()
        self.target_colm = target_colm

        self.default_window_size = len(self.data)/10
        self.default_step_size = horizon_length

        """
            Initialize timesfm model with the following parameters:
            content_len: The max length of context on which the predictions will be done | Currently supports a max of 512
            horizon_len: The number of future days for which the predictions will be made | reccomended horizon length <= context length
            input_patch_len: Fixed value to load 200m model
            output_patch_len: Fixed value to load 200m model
            num_layers: Fixed value to load 200m model
            model_dims: Fixed value to load 200m model
            backend: The backend to be used for the model | Currently supports "cpu", "gpu", "tpu"
        """
        self.tfm = timesfm.TimesFm(
            content_len = min(content_length, 512), 
            horizon_len=horizon_length,             
            input_patch_len=32,                     
            output_patch_len=128,                   
            num_layers=20,                          
            model_dims=1280,                        
            backend=backend,                        
        )
        
        logging.info("Loading model from checkpoint")
        self.tfm.load_from_checkpoint(repo_id=checkpoint)
        
        logging.info("Model loaded successfully")

    
    def data_preprocess(self, date_colm: str) -> None:
        self.data = self.data.astype({date_colm: 'datetime64[ns]', self.target_colm: float})

    
    def _iter_split(self, current_window: int, step_size: int):
        window_data = self.data[:current_window]
        
        if current_window + step_size > len(self.data):
            step_size = len(self.data) - current_window
        
        return window_data, step_size


    def predict(self, intial_window_size: int = None, step_size: int = None, freq: str = "D"):
        
        logging.info("Starting predictions")
        
        initial_window_size = initial_window_size or self.default_window_size
        step_size = step_size or self.default_step_size
        
        logging.info(f"Initial window size: {initial_window_size}")
        logging.info(f"Step size: {step_size}")
        
        
        # Run iterations and return a pd series of predictions
        self.data["unique_id"] = 0
        window = intial_window_size
        predictions = pd.Series()
        
        while window < len(self.data):
            logging.info(f"Predicting for window size: {window}")
            current_window, step_size = self._iter_split(window, step_size)
            batch_pred = self.tfm.forecast_on_df(current_window, freq=freq, value_name=self.target_colm)['timesfm']
            predictions = pd.concat([predictions, batch_pred])
            window += step_size
        
        logging.debug(f"Buffer: {len(predictions) - (window - initial_window_size)}")
        predictions = predictions[:-(len(predictions) - (window - initial_window_size))]
        predictions.index = range(initial_window_size, window)
        return predictions
            
        
        



