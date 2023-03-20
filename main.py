if __name__ == "__main__":
    import cv2
    import numpy as np
    import os
    import time
    import matplotlib.pyplot as plt
    import json
    import sys

    from BufferlessVideoCapture import BufferlessVideoCapture
    from Functions import get_circle_triangle
    from Robot import Robot

    # Check user specified the environment the robot is in, and what system it is being run on
    if len(sys.argv) != 3:
        print("Please enter the environment you're running in (home, idp_1, idp_2) and whether you are on a laptop SSHed in, or the IDP PC (SSH, IDP)")
        quit()

    # Extract the environment and system type from the command line arguements
    environ = sys.argv[1]

    # SSH will be recognised in upper or lower case, anything else defaults to being on an IDP PC
    is_ssh = sys.argv[2].lower() == "ssh"

    # Load the config file containing all environment configs
    with open("environ_configs.json", "rb") as f:
        environ_configs = json.load(f)

    # Check the specified environment is in the config file
    if environ in environ_configs.keys():
        # Extract the relevant config info
        config = environ_configs[environ]
    else:
        print("Environment not found in config file")
        quit()


    # Set up video feed

    # Get the correct identifier / url based on what system is being used
    if is_ssh:
        camera_indentifier = config["camera_identifier_ssh"]
    else:
        camera_indentifier = config["camera_identifier_idp"]


    # Using dshow greatly speeds up initialisation time for local cameras, but isn't compatible with network cameras, so we select whether to use it or not

    if config["use_dshow"]:
        video_cap = cv2.VideoCapture(camera_indentifier, cv2.CAP_DSHOW)
    else:
        video_cap = cv2.VideoCapture(camera_indentifier)

    video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    # Set the max buffer size for the camera feed

    # Get a frame from the camera to analise it's dimensions
    ret, frame = video_cap.read()
    print("Camera feed dimensions")
    print(frame.shape)


    # Specifiy that we want every x pixels, e.g. being set to 2 would get every other pixel, in both dimensions
    SKIP_AMOUNT = 2

    full_width = frame.shape[1]
    full_height = frame.shape[0]

    # Both crop and sub-sample the full frame, cropping is done to remove any unnecessary parts of the image around the table
    sub_frame = frame[config["top_frame_cutoff"]:full_height - config["bottom_frame_cutoff"]:SKIP_AMOUNT,
                config["left_frame_cutoff"]:full_width - config["right_frame_cutoff"]:SKIP_AMOUNT, ::]

    # Record the new / cropped-sampled dimensions
    cropped_width = sub_frame.shape[1]
    cropped_height = sub_frame.shape[0]
    print(f"Running at {cropped_width}x{cropped_height} (wxh)")


    # Initialise an instance of the Robot class which handles all the calculations for positions, motor speeds etc
    robot = Robot()

    # Run the main loop
    while True:
        # Get the current frame
        ret, frame = video_cap.read()

        # Check there is actually a frame
        if frame is not None:
            # If there is a frame, proceed with analysis

            # Crop and subsample the frame as required / done before
            frame = frame[config["top_frame_cutoff"]:full_height - config["bottom_frame_cutoff"]:SKIP_AMOUNT,
                    config["left_frame_cutoff"]:full_width - config["right_frame_cutoff"]:SKIP_AMOUNT]

            # Record a copy of the full colour frame before it is converted to grayscale, which is used to check if points are pink or not
            colour_frame = frame
            gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Identify the coordinates of the circles drawn on top of the robot (in a triangle)
            # Returns whether it found the circles, and if so, the first coordinate is of the biggest circle (i.e. the forwards one)
            circle_params = (
                config["circle_min_dist"],
                config["circle_min_radius"], config["circle_max_radius"], config["circle_param_1"], config["circle_param_2"]
                )
                
            ellipse_params = (
                config["ellipse_min_area"], config["ellipse_max_area"], config["ellipse_min_convexity"], config["ellipse_min_inertia"],
                config["ellipse_triangle_min_side_length"], config["ellipse_triangle_max_side_length"]
            )

            both_params = (
                config["circle_colour_detection_ratios"][0], config["circle_colour_detection_ratios"][1], config["circle_triangle_error_threshold"]
            )

            tri_found, coords = get_circle_triangle(gray_scale_frame, colour_frame, circle_params, ellipse_params, both_params, SKIP_AMOUNT)


            # If desired, display the frame on the screen, runs much faster if not, so used for debugging purposes
            display = True
            
            # Plot the points found on the image, if they are found
            if tri_found:
                robot.update(coords)
                if display:
                    for i, p in enumerate(coords):
                        # Plot the circles, with the front one a different colour
                        gray_scale_frame = cv2.circle(gray_scale_frame, (int(p[0]), int(p[1])), 10 // (SKIP_AMOUNT), [255, 0, 0][i % 3], -1)

            if display: ## Draw angle vector and robot position
                gray_scale_frame = cv2.circle(gray_scale_frame, robot.position, 5, 0, -1)

                dir_line_length = 30
                dir_end_point = (int(robot.position[0] + (dir_line_length * np.cos(robot.rotation))), int(robot.position[1] + (dir_line_length * np.sin(robot.rotation))))
                gray_scale_frame = cv2.line(gray_scale_frame, robot.position, dir_end_point, 255, thickness=2)
            if display: # Draw target on picture
                gray_scale_frame = cv2.circle(gray_scale_frame, robot.current_target, 10, 0, 2)
            if display:
                cv2.imshow('frame',gray_scale_frame)

                # Press q to close the video and exit the main loop
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

        else:
            print("Frame is None")
            break


    # When everything done, release the capture
    video_cap.release()
    cv2.destroyAllWindows()
    print("Video stop")
    robot.stop()