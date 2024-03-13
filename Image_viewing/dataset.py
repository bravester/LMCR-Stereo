import os
import re
import cv2
import numpy as np


class PathNotFound(Exception):
    """Path not found exception"""
    def __init__(self, filepath, message="Path does not exist: "):
        """
        Exception handelling for path not found
        Parameters:
            filepath (string): Filepath that was not found.
                Will be displayed in exception message
            message (string): Message to display in exception,
                will use default message if no message is provided
        """
        self.filepath = filepath
        self.message = message
        super().__init__(self.message)

    @staticmethod
    def validate(filepath):
        """
        Validate standard exception condition.
        Raises exception if validation fails.

        Parameters:
            filepath (string): filepath to test
        """
        if not os.path.exists(filepath):
            raise PathNotFound(filepath)

    def __str__(self):
        """Overload of exception message"""
        return self.message+self.filepath

class MalformedPFM(Exception):
    """Malformed PFM file exception"""
    def __init__(self, message="Malformed PFM file"):
        """
        Exception handelling for malformed PFM file
        Parameters:
            message (string): Message to display in exception,
                will use default message if no message is provided
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """Overload of exception message"""
        return self.message


def load_pfm(filepath):
    """
    Load pfm data from file

    Parameters:
        filepath (string): filepath to pfm image (e.g. image.pfm)

    Returns:
        pfm_data (numpy): 2D image filled with data from pfm file
    """
    # Check file exists
    if not os.path.exists(filepath):
        raise PathNotFound(filepath, "Pfm file does not exist")
    # Open pfm file
    file = open(filepath, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    # Read header to check pf type
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise MalformedPFM('Not a PFM file.')

    # Read dimensions from pfm file and check they match expected：从pfm文件读取尺寸，并检查它们是否符合预期
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise MalformedPFM('Malformed PFM header')

    # Read data scale from file：从文件中读取数据的尺寸
    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    # Read image data from pfm file：从pfm文件中读到图像数据
    data = np.fromfile(file, endian + 'f')
    # Format image data into expected numpy image
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    # Flip image vertically as image appears upside-down：当图像上下颠倒时垂直翻转图像
    data = cv2.flip(data, 0)
    return data, scale