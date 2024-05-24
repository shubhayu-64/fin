import pandas as pd
import numpy as np
import timesfm


class Predictions:
    def __init__(self, 
                 data: pd.DataFrame, 
                 target_colm: str,
                 content_length: int = 512, 
                 horizon_length: int = 14, 
                 backend: str = "cpu", 
                 checkpoint: str = "google/timesfm-1.0-200m",
                 ) -> None:
        
        self.data = data.copy()
        self.target_colm = target_colm

        self.initial_window_size = len(self.data)/10
        self.step_size = horizon_length

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

        self.tfm.load_from_checkpoint(repo_id=checkpoint)

    
    def data_preprocess(self):
        # Cleanup the data according to the time series variable column
        pass

    
    def _iter_split(self):
        # Iteratively split the data by step size 
        pass


    def predict(self):
        # Run iterations and return a pd series of predictions
        pass



