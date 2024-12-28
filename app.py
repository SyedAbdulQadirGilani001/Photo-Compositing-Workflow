import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to load and resize images
def load_image(image_path, size=(500, 500)):
    """
    Loads an image from the provided path and resizes it to the given size.
    Args:
        image_path (str): Path to the image file.
        size (tuple): The target size (width, height) for resizing.
    Returns:
        image (ndarray): Resized image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, size)  # Resize image to the specified size
    return image

# Function to segment a person from the background using color range
def segment_person(image):
    """
    Segments the person from the background using color thresholds.
    Assumes a relatively simple background.
    Args:
        image (ndarray): The input image containing the person.
    Returns:
        result (ndarray): The segmented person with transparent background.
    """
    lower = np.array([35, 35, 35])  # Lower bound of background color (green)
    upper = np.array([85, 85, 85])  # Upper bound of background color (green)
    
    # Create a mask where the background is detected
    mask = cv2.inRange(image, lower, upper)
    
    # Create an inverse mask to extract the person
    inverse_mask = cv2.bitwise_not(mask)
    
    # Extract the person using the inverse mask
    result = cv2.bitwise_and(image, image, mask=inverse_mask)
    return result

# Function to overlay person images on the background
def create_group_photo(people_images, background_image_path, positions):
    """
    Creates a group photo by overlaying people images on a background.
    Args:
        people_images (list): List of segmented people images.
        background_image_path (str): Path to the background image.
        positions (list): List of positions for each person in the photo.
    Returns:
        result (ndarray): The generated group photo.
    """
    # Load the background image
    background = load_image(background_image_path, size=(1200, 800))  # Example size
    
    # Place each person at the corresponding position
    for idx, img in enumerate(people_images):
        x_offset, y_offset = positions[idx]
        
        h, w = img.shape[:2]
        background[y_offset:y_offset+h, x_offset:x_offset+w] = img[:, :, :3]  # Only RGB channels
        
    return background

# Streamlit UI to upload images and generate the group photo
def main():
    st.title("Group Photo Generator")
    
    # Upload background image
    background_image = st.file_uploader("Upload Background Image", type=["jpg", "png"])
    if background_image is not None:
        background_path = f"background.{background_image.name.split('.')[-1]}"
        with open(background_path, "wb") as f:
            f.write(background_image.getbuffer())
        
        st.image(background_image, caption="Background Image", use_column_width=True)
    
    # Upload images of people (max 10 people)
    st.subheader("Upload People Images")
    people_images = []
    for i in range(1, 11):
        person_image = st.file_uploader(f"Upload Person {i}", type=["jpg", "png"])
        if person_image is not None:
            person_path = f"person{i}.{person_image.name.split('.')[-1]}"
            with open(person_path, "wb") as f:
                f.write(person_image.getbuffer())
            person_img = load_image(person_path)  # Load and resize the image
            segmented_person = segment_person(person_img)  # Segment the person
            people_images.append(segmented_person)  # Add to list
    
    if len(people_images) > 0 and background_image is not None:
        # Example positions for each person (You can dynamically set these)
        positions = [(100, 100), (300, 150), (500, 200), (700, 250), 
                     (900, 300), (1100, 350), (1300, 400), (1500, 450),
                     (1700, 500), (1900, 550)]
        
        # Generate group photo
        group_photo = create_group_photo(people_images, background_path, positions)
        
        # Display the generated photo
        st.image(group_photo, caption="Generated Group Photo", use_column_width=True)
        
        # Option to download the generated image
        result_path = "generated_group_photo.jpg"
        cv2.imwrite(result_path, cv2.cvtColor(group_photo, cv2.COLOR_RGB2BGR))
        st.download_button(
            label="Download Group Photo",
            data=open(result_path, "rb").read(),
            file_name="group_photo.jpg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()
