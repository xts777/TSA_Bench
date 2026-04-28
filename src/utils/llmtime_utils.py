import numpy as np
import pandas as pd

class LLMTimeSerializer:
    """
    Handles serialization and deserialization of time-series data for LLMs.
    Based on the methodology from ngruver/llmtime.
    """
    def __init__(self, precision=2, separator=' ', bit_sep=''):
        self.precision = precision
        self.separator = separator
        self.bit_sep = bit_sep

    def serialize(self, data):
        """
        Converts numerical data into a string format suitable for LLMs.
        Example: 123.45 -> '1 2 3 . 4 5'
        """
        if isinstance(data, (np.ndarray, list)):
            return self.separator.join([self._serialize_single(x) for x in data])
        return self._serialize_single(data)

    def _serialize_single(self, x):
        # Format to fixed precision
        s = format(x, f'.{self.precision}f')
        # Add spaces between characters
        return self.bit_sep.join(list(s))

    def deserialize(self, s):
        """
        Converts LLM output strings back into numerical values.
        """
        # Remove spaces and try to convert to float
        clean_s = s.replace(self.bit_sep, '').replace(self.separator, ' ')
        parts = clean_s.split()
        values = []
        for p in parts:
            try:
                values.append(float(p))
            except ValueError:
                continue
        return np.array(values)

class LLMQuantileScaler:
    """
    Scales data using quantiles to a range suitable for serialization.
    """
    def __init__(self, out_range=(0, 1)):
        self.out_range = out_range
        self.quantiles = None
        self.references = None

    def fit(self, data):
        self.references = np.sort(data.flatten())
        self.quantiles = np.linspace(0, 1, len(self.references))

    def transform(self, data):
        if self.references is None:
            raise ValueError("Scaler must be fitted before transform.")
        
        # Map data to [0, 1] based on quantiles
        scaled = np.interp(data, self.references, self.quantiles)
        # Rescale to out_range
        scaled = scaled * (self.out_range[1] - self.out_range[0]) + self.out_range[0]
        return scaled

    def inverse_transform(self, data):
        if self.references is None:
            raise ValueError("Scaler must be fitted before inverse_transform.")
        
        # Map [out_range] back to [0, 1]
        scaled = (data - self.out_range[0]) / (self.out_range[1] - self.out_range[0])
        # Map back to original values
        return np.interp(scaled, self.quantiles, self.references)
