"""
Design parameters class for managing metamaterial design configurations.

This module provides a class for handling design parameters and their
associated properties for 2D metamaterial structures.
"""

import numpy as np


class DesignParameters:
    """
    Class for managing design parameters of metamaterial structures.
    
    This class handles the configuration of design parameters including
    property coupling, design style, and various design options.
    """
    
    def __init__(self, design_number):
        """
        Initialize design parameters.
        
        Parameters
        ----------
        design_number : int or array_like
            Design number(s) for the structure
        """
        self.property_coupling = 'coupled'
        self.design_number = design_number
        self.design_style = 'matern52'
        self.design_options = {
            'sigma_f': 1.0,
            'sigma_l': 0.5,
            'symmetry': 'none',
            'N_value': 3
        }
        self.N_pix = [5, 5]
    
    def prepare(self):
        """
        Prepare the design parameters by expanding property information.
        
        Returns
        -------
        self : DesignParameters
            Updated design parameters object
        """
        return self.expand_property_information()
    
    def expand_property_information(self):
        """
        Expand property information to ensure consistent dimensions.
        
        Returns
        -------
        self : DesignParameters
            Updated design parameters object
        """
        # Expand design_number
        if np.isscalar(self.design_number):
            self.design_number = np.full(3, self.design_number)
        
        # Expand design_style
        if isinstance(self.design_style, str):
            temp = self.design_style
            self.design_style = [temp] * 3
        elif isinstance(self.design_style, list):
            if len(self.design_style) == 1:
                self.design_style = self.design_style * 3
        
        # Expand design_options
        if isinstance(self.design_options, dict):
            temp = self.design_options.copy()
            self.design_options = [temp] * 3
        
        return self

