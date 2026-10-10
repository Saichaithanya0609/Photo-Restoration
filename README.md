# Photo Restoration and Enhancement System

A Digital Image Processing (DIP) project focused on improving the visual quality of old photographs using image-processing techniques.

## Project Overview

Old photographs may suffer from fading, low contrast, noise, and loss of sharpness. This project aims to provide a simple interface where users can upload an image, enhance its quality using DIP algorithms, compare the original and processed images, and download the result.

## Current Progress

### Frontend

The frontend is built using React and Vite.

Completed so far:

- Basic website layout and navigation.
- Home page with project introduction.
- About, Features, and Our Work pages.
- Image upload interface.
- Original image preview.
- Enhancement settings interface for contrast, noise reduction, and sharpening.
- Basic image download functionality for the original uploaded image.
- Responsive styling for the current interface.

**Current status:** The frontend prototype is in progress. The image-processing functionality has not yet been connected to the backend.

## Planned Technologies

- **Frontend:** React, Vite, JavaScript, CSS
- **Backend:** Python, FastAPI
- **Image Processing:** OpenCV, NumPy
- **Version Control:** Git and GitHub

## Planned Image Processing Techniques

The following techniques are planned for implementation and testing:

- **CLAHE:** To improve local contrast in faded or low-contrast photographs.
- **Median Filtering:** To reduce suitable types of image noise.
- **Image Sharpening:** To enhance edges and improve perceived clarity.
- **Image Inpainting (optional):** To explore the removal of scratches or damaged regions if time permits.

These techniques will be evaluated on sample images before being integrated into the application.

## Project Workflow

The intended workflow is:

1. Upload an old photograph.
2. Preview the original image.
3. Select enhancement settings.
4. Send the image to the FastAPI backend.
5. Process the image using OpenCV and NumPy.
6. Display the enhanced result alongside the original.
7. Download the processed image.

Steps involving backend processing and enhanced-image output are planned and are not implemented yet.

## Getting Started

### Prerequisites

- Node.js and npm
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/Saichaithanya0609/Photo-Restoration.git
```

Navigate to the frontend directory:

```bash
cd Photo-Restoration/Frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed in your terminal.

## Team Collaboration

The project is maintained on GitHub.

- `main`: Intended for stable, reviewed code.
- `testing`: Shared branch for development and testing.

Contributors should coordinate changes and avoid overwriting one another's work. Changes should be reviewed before being merged into `main`.

## Future Work

- Set up the FastAPI backend.
- Implement and test OpenCV image-processing algorithms.
- Connect the React frontend to the backend.
- Display original and enhanced images for comparison.
- Implement downloading of processed images.
- Test the application using different types of old and damaged photographs.
- Document results and limitations.

## Project Status

**Current stage:** React frontend prototype and GitHub collaboration setup.

The backend and actual image-enhancement algorithms are planned next.
