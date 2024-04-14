import cv2

def extract_signatures(image_path, output_folder):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours
    for i, contour in enumerate(contours):
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the signature from the image
        signature = image[y:y+h, x:x+w]
        
        # Save the cropped signature
        cv2.imwrite(f"{output_folder}/signature_{i}.jpg", signature)
        print(f"Signature {i+1} extracted and saved.")
        
if __name__ == "__main__":
    # Path to the input image
    input_image_path = r"C:\Users\Bohara\Downloads\IMG_4457.jpg"
    
    # Output folder to save extracted signatures
    output_folder = r"C:\Users\Bohara\Downloads\signature_extraction"
    
    # Create the output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Call the function to extract signatures
    extract_signatures(input_image_path, output_folder)
