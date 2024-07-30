import numpy as np
import torch
import torchvision

from typing import Any, Optional, Tuple, Generic, TypeVar
from collections import namedtuple

BBox = namedtuple("BBox", "x_min y_min width height")
Shape = TypeVar("Shape")
DType = TypeVar("DType")

class Frame(np.ndarray, Generic[Shape, DType]):
    pass



class Mosse:
    def __init__(self, frame_shape: Tuple[int], sigma: float=1.0, num_perturbations: int=256, min_psr: float=8.0, seed: int=2137) -> None:
        """
        Constructor.

        Args:
            frame_shape (Tuple[int]): input frame dimensions
            sigma (float, optional): used in Gaussian distribution. Defaults to 1.0.
            num_perturbations (int, optional): number of rotations of input frame in init. Defaults to 128.
            min_psr (float, optional): minimum acceptable value of PSR. Defaults to 8.0.
            seed (int, optional): seed for torch. Defaults to 2137.
        """
        self._sigma = sigma
        self._num_perturbations = num_perturbations
        self._min_psr = min_psr
        self._bbox_xywh = None
        self._device = "cpu"
        torch.manual_seed(seed)
    
    def init(self, frame: Frame[Tuple[int, int], np.uint8], bbox_xywh) -> bool:
        """
        Initializes the MOSSE filter.
        Computes the desired output Gi (Gaussian distribution with the peak centered on the target), preprocesses the area of interest,
        rotates it to create the training input fi, and computes the numerator and denominator of the objective function to be minimized.

        Args:
            frame (Frame[Tuple[int, int], np.uint8]): input frame
            bbox_xywh (_type_): bounding box (named tuple containing dimensions)

        Returns:
            bool: True upon success
        """
        
        assert frame.ndim == 2 and frame.dtype == np.uint8, "invalid frame"
        
        self._bbox_xywh = BBox(*bbox_xywh)
        assert self._bbox_xywh.width > 0 and self._bbox_xywh.height > 0, "invalid bounding box"
        x_min, y_min, bbox_width, bbox_height = bbox_xywh
        
        # calculate desired output of convolution Gi in Fourier domain
        self._Gi = torch.fft.fft2(self._get_gauss(self._bbox_xywh.width, self._bbox_xywh.height))
        roi = frame[y_min : y_min + bbox_height, x_min : x_min + bbox_width] # select region of interest
        roi = self._preprocess(roi) # preprocess the region of interest
        fi = self._perturbate_frame(roi, self._num_perturbations) # create the training input set by rotating the frame
        Fi = torch.fft.fft2(fi) # switch to Fourier domain
        Fi_conj = torch.conj(Fi) # calculate conjugate
        # calculate the numerator and denominator of the objective function to be minimized
        Num = self._Gi[None, :, :].repeat((self._num_perturbations, 1, 1)) * Fi_conj # correlation between input and desired output
        Denom = Fi * Fi_conj # energy spectrum of the input
        
        self._Num = torch.sum(Num, dim=0)
        self._Denom = torch.sum(Denom, dim=0)
        
        return True
     
        
    def update(self, frame: Frame[Tuple[int, int], np.uint8], rate: float=0.07, eps: float=1e-5) -> BBox:
        """
        Calculate the response, update bounding box.
        Computes the MOSSE filter response, calculates the PSR, checks for changes in bounding box dimensions,
        and updates the bounding box.

        Args:
            frame (np.array[Tuple[int, int]]): input frame
            rate (float, optional): rate at which the new frame impacts the filter. Defaults to 0.125.
            eps (float, optional): epsilor for PSR calculation. Defaults to 1e-5.

        Returns:
            BBox: new bounding box
        """
        assert frame.ndim == 2 and frame.dtype == np.uint8, "invalid frame"
        x_min, y_min, bbox_width, bbox_height = self._bbox_xywh
        
        roi = frame[y_min : y_min + bbox_height, x_min : x_min + bbox_width]
        fi = self._preprocess(roi) # preprocess the input
        Fi = torch.fft.fft2(fi) # switch to Fourier domain
        Hi = self._Num / self._Denom # calculate the filter
        # calculate the tracker response, inverse FFT
        response = torch.fft.ifft2(Hi * Fi).real # correlation is element-wise multiplication in Fourier domain

        # update the bounding box
        new_row_center, new_col_center = (response == torch.max(response)).nonzero().to("cpu").detach().numpy()[0]
        dx = int(new_col_center - bbox_width / 2.0)
        dy = int(new_row_center - bbox_height / 2.0)
        new_bbox = Mosse.correct_bbox(BBox(x_min + dx, y_min + dy, bbox_width, bbox_height), frame.shape[1], frame.shape[0])
        if new_bbox.width != self._bbox_xywh.width or new_bbox.height != self._bbox_xywh.height:
            self.init(frame, new_bbox)
        else:
            self._update_filter(frame, self._bbox_xywh, rate)
        
        #update bounding box    
        self._bbox_xywh = new_bbox
        
        # calculate PSR
        psr = (response[new_row_center, new_col_center] - response.mean()) / (response.std() + eps)
        
        return self._bbox_xywh 
           
        
    def _get_gauss(self, width: int, height: int) -> torch.Tensor:
        """
        Creates a 2D Gaussian distribution with the peak on the target center.

        Args:
            width (int): bounding box width
            height (int): bounding box height

        Returns:
            torch.Tensor: 2D Gaussian distribution
        """
        x_center = width / 2
        y_center = height / 2
        yy, xx = torch.meshgrid(torch.arange(height, device=self._device), torch.arange(width, device=self._device))
        gauss = (torch.square(xx - x_center) + torch.square(yy - y_center)) / (2 * self._sigma)
        gauss = torch.exp(-gauss)
        gauss -= gauss.min()
        gauss /= gauss.max() - gauss.min()
        
        return gauss
    
    
    def _update_filter(self, frame: Frame[Tuple[int, int], np.uint8], bbox_xywh: BBox, rate: float) -> None:
        """
        Updates the objective function.

        Args:
            frame (Frame[Tuple[int, int], np.uint8]): input frame
            bbox_xywh (BBox): bounding box dimensions
            rate (float): rate at which the new frame impacts the filter
        """
        x_min, y_min, bbox_width, bbox_height = bbox_xywh
        roi = frame[y_min : y_min + bbox_height, x_min : x_min + bbox_width]
        fi = self._preprocess(roi)
        Fi = torch.fft.fft2(fi)
        Fi_conj = torch.conj(Fi)
        # update filter with new frame
        # self._Num = rate * self._Gi * Fi_conj + (1 - rate) * self._Num
        # self._Denom = rate * Fi * Fi_conj + (1 - rate) * self._Denom
        self._Num = self._Num + self._Gi * Fi_conj
        self._Denom = self._Denom + Fi * Fi_conj
        
    
    def _perturbate_frame(self, frame: torch.Tensor, num_samples: int, degree: float=18.0) -> torch.Tensor:
        """
        Rotates the input frame to create the learning input.

        Args:
            frame (torch.Tensor): input frame
            num_samples (int): number of rotations
            degree (float, optional): degree of rotation. Defaults to 18.0.

        Returns:
            torch.Tensor: output frames
        """
        transform = torchvision.transforms.RandomRotation(degree)
        
        return transform(frame[None, :, :].repeat((num_samples, 1, 1)))
        
    
    def _preprocess(self, frame: Frame[Tuple[int, int], np.uint8], eps: float=1e-5) -> torch.Tensor:
        """
        Preprocesses the frame.
        Transforms pixel values using log function. Normalizes the pixel values (mean of 0.0, norm of 1.0), multiplies by
        Hann window.

        Args:
            frame (Frame[Tuple[int, int], np.uint8]): input frame
            eps (float, optional): epsilon for normalization. Defaults to 1e-5.

        Returns:
            torch.Tensor: normalized frame
        """
        frame_tensor = torch.from_numpy(frame.astype(np.float32)).to(self._device) # convert to tensor
        frame_tensor = torch.log(frame_tensor + 1) # perform log function
        frame_tensor = (frame_tensor - frame_tensor.mean()) / (frame_tensor + eps) # mean 0.0, norm 1.0
        height, width = frame.shape[:2]
        
        return frame_tensor * Mosse.hann(height, width, self._device)
        
    
    @staticmethod
    def hann(height: int, width: int, device: torch.device=torch.device("cpu")) -> torch.Tensor:
        """
        Calculates the Hann window.

        Args:
            height (int): frame height
            width (int): frame width
            device (torch.device, optional): cpu or gpu. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: frame after Han smoothing
        """
        row_mask, col_mask = torch.meshgrid(torch.hann_window(height, device=device), torch.hann_window(width, device=device))
        
        return row_mask * col_mask
    
    
    @staticmethod
    def correct_bbox(bbox_xywh: BBox, width: int, height: int) -> BBox:
        """
        Corrects the bounding box.
        Checks for negative values and frame edges.

        Args:
            bbox_xywh (BBox): input bounding box
            height (int): frame height
            width (int): frame width

        Returns:
            BBox: correct bounding box
        """
        x_min, y_min, bbox_width, bbox_height = bbox_xywh
        x_max = x_min + bbox_width
        y_max = y_min + bbox_height
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width - 1, x_max)
        y_max = min(height - 1, y_max)
        
        return BBox(x_min, y_min, x_max - x_min, y_max - y_min)