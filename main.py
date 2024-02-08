import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import  os
from roboflow import Roboflow
def main():
    # Sidebar with project information
    st.sidebar.title("About this Project")
    st.sidebar.write("This is a Streamlit app for ........")
    st.sidebar.write("Feel free to upload images and check the volume of drusen in it.")

    st.title("Drusen Segmentation App")
    st.write("Upload an image and check detect the drusen.")

    def process_image(image_path):
        rf = Roboflow(api_key="3C3rSc4RNS6DFKkOvQwY")
        project = rf.workspace().project("drusen-o7wva")
        model = project.version(1).model

        model.predict(image_path).save("prediction.jpg")
        result = model.predict(image_path, confidence=10).json()

        # Load the image
        img = plt.imread(image_path)

        # Create a figure and axis
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Extract the points from the prediction result and add polygons to the axis
        predictions = result['predictions']
        for prediction in predictions:
            points = prediction['points']
            x_values = [point['x'] for point in points]
            y_values = [point['y'] for point in points]

            # Create a polygon patch using the extracted points
            polygon = patches.Polygon(list(zip(x_values, y_values)), closed=True, fill=None, edgecolor='r')
            ax.add_patch(polygon)

        # Set axis limits
        width = int(result['image']['width'])
        height = int(result['image']['height'])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        # Show the plot
        plt.axis('off')
        st.pyplot(fig)

    # Streamlit app
    def main():
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            image_path = "temp_image.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the image and display the result
            process_image(image_path)

            # Remove the temporary image file
            os.remove(image_path)

    if __name__ == "__main__":
        main()
    # Owner and GitHub info in Markdown
    st.markdown("""
    ## About the Owner
    - Name: Moshood Abubakar
    - Role: Data Scientist
    - Location: Bradford, United Kingdom

    ## GitHub Repository
    - [GitHub](https://github.com/yourusername)
    """)

main()