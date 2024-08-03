# Fingerprint Image Enhancement System

This project is designed to enhance fingerprint images using Python and OpenCV. The enhancement process includes ridge segmentation, ridge orientation estimation, Gabor filtering, binarization, and thinning.

## Features

- **Ridge Segmentation:** Segments the ridges in a fingerprint image using local variance thresholding.
- **Ridge Orientation Estimation:** Estimates ridge orientations using a gradient-based approach.
- **Gabor Filtering:** Applies Gabor filters with various orientations and frequencies to enhance the fingerprint image.
- **Binarization:** Converts the enhanced image to a binary format using a specified threshold.
- **Thinning:** Applies morphological thinning to reduce ridge thickness and improve fingerprint clarity.

## Installation

To get started with the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary Python libraries using the provided `requirements.txt` file.

## Usage

1. Place your fingerprint image in the project directory with the filename `fingerprint.jpg`.
2. Run the Python script to process the image. The script will perform the following steps:
   - Enhance the fingerprint image.
   - Binarize the enhanced image.
   - Apply thinning to the binarized image.

3. The script will display four images:
   - **Original Image:** The input fingerprint image.
   - **Enhanced Image:** The result after applying enhancement techniques.
   - **Binarized Image:** The binary version of the enhanced image.
   - **Thinned Image:** The final thinned version of the binarized image.

## Code Overview

The project includes functions to:
- Segment ridges from the fingerprint image.
- Estimate ridge orientations.
- Enhance the image using Gabor filters.
- Binarize the image.
- Apply thinning to the binarized image.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- Special thanks to the OpenCV library for providing the computer vision functionalities.
- Thanks to NumPy and SciPy for their numerical and scientific computation capabilities.
