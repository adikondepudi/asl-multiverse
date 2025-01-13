import numpy as np
from typing import Dict, Optional

class ASLDataLoader:
    """Base class for ASL data loading"""
    def load_data(self, filepath: str) -> np.ndarray:
        """Load data from file"""
        raise NotImplementedError()
    
    def save_data(self, filepath: str, data: np.ndarray) -> None:
        """Save data to file"""
        raise NotImplementedError()

class NiftiLoader(ASLDataLoader):
    """NIFTI file loader implementation"""
    def __init__(self):
        try:
            import nibabel as nib
            self.nib = nib
        except ImportError:
            raise ImportError("nibabel is required for NiftiLoader")
    
    def load_data(self, filepath: str) -> np.ndarray:
        """Load data from NIFTI file"""
        img = self.nib.load(filepath)
        data = img.get_fdata()
        return data
    
    def save_data(self, filepath: str, data: np.ndarray) -> None:
        """Save data to NIFTI file"""
        img = self.nib.Nifti1Image(data, np.eye(4))
        self.nib.save(img, filepath)

class MockLoader(ASLDataLoader):
    """Mock loader for testing"""
    def __init__(self, mock_data: Optional[np.ndarray] = None):
        self.mock_data = mock_data
    
    def load_data(self, filepath: str) -> np.ndarray:
        """Return mock data or generate synthetic data"""
        if self.mock_data is not None:
            return self.mock_data
            
        # Generate synthetic data if none provided
        nx, ny, nz = 64, 64, 20
        data = np.random.normal(100, 10, (nx, ny, nz, 2))
        data[:,:,:,1] = data[:,:,:,0] - 2  # Add controlled difference
        return data
    
    def save_data(self, filepath: str, data: np.ndarray) -> None:
        """Mock save operation"""
        self.mock_data = data