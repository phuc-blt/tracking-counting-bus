import cv2
from tracking import ObjectCounter  # Importing ObjectCounter from tracker.py

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Check for mouse movement
        point = [x, y]
        print(f"Mouse moved to: {point}")

# Open the video file
cap = cv2.VideoCapture('/workspace/phucnt/bus/data/video.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define region points for counting
region_points = [(386, 103), (200, 300)]

# Initialize the object counter
counter = ObjectCounter(
    region=region_points,  # Pass region points
    model="yolo11x.pt",  # Model for object counting
    classes=[0],  # Detect only person class
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust line width for display
)

# Create a named window and set the mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (1020, 500))

count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
        # If video ends, reset to the beginning
#        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#        continue
    count += 1
    if count % 2 != 0:  # Skip odd frames
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Process the frame with the object counter
    frame = counter.count(frame)

    # Show the frame
    cv2.imshow("RGB", frame)
    
    # Write the frame to the output video file
    out.write(frame)

    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break

# Release the video capture and writer objects, and close display windows
cap.release()
out.release()
cv2.destroyAllWindows()
