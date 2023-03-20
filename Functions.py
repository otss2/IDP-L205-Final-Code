import cv2
import numpy as np
import time
import itertools

def get_circle_triangle(gray_img, colour_img, circle_params, ellipse_params, both_params, SKIP_AMOUNT):
    # Blur the image to improve circle detection accuracy
    gray_img = cv2.blur(gray_img, (3, 3))

    # Get params that are used in both the circle detection or ellipse detection
    RG_ratio, BG_ratio, error_thresh = both_params

    # Select whether doing circles or ellipses
    do_circles = False
    if do_circles:
        # Run circle detection algorithm

        min_dist, minR, maxR, param1, param2  = circle_params

        """
        Param 1 will set the sensitivity; how strong the edges of the circles need to be.
        Too high and it won't detect anything, too low and it will find too much clutter.
        Param 2 will set how many edge points it needs to find to declare that it's found a circle.
        Again, too high will detect nothing, too low will declare anything to be a circle.
        The ideal value of param 2 will be related to the circumference of the circles.
        """

        # Run the detection, returns the centre and radius of the circles
        detected_circles = cv2.HoughCircles(gray_img,
                    cv2.HOUGH_GRADIENT, 1, min_dist, param1 = param1,
                param2 = param2, minRadius = minR, maxRadius = maxR)

        # Returns with an extra dimension so remove that
        if detected_circles is not None:
            detected_circles = detected_circles[0]

        # Takes ~ 15ms

    else:
        # Ellipse detection

        # Set the parameters for the blob detector (filters for blobs which are ellipse like)
        min_area, max_area, min_convex, min_inertia, min_side_length, max_side_length = ellipse_params

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = min_area / (SKIP_AMOUNT ** 2)
        params.maxArea = max_area / (SKIP_AMOUNT ** 2)

        params.filterByConvexity = True
        params.minConvexity = min_convex

        params.filterByInertia = True
        params.minInertiaRatio = min_inertia

        detector = cv2.SimpleBlobDetector_create(params)

        # Run the detection and extract the centre and size of each ellipse
        detected_circles = [(k.pt[0], k.pt[1], k.size) for k in detector.detect(gray_img)]
        # Takes ~ 45ms on full-res, ~15ms on half-res

    # Check at least 1 circle / ellipse detected, if not, return False and None for detected and coords
    if detected_circles is not None:

        # Filter out any circles / ellipses that are centered outside the image
        filtered_circles = []
        for circle in detected_circles:
            if (not 0 <= circle[0] <= gray_img.shape[0]) or (not 0 <= circle[1] <= gray_img.shape[1]):
                pass
            else:
                filtered_circles.append(circle)

        # Filter for pink circles / ellipses and store in coloured_circles
        # RGB is 255 0 255 for pink
        coloured_circles = []
        for circle in filtered_circles:
            # Extract the colour at the circle / ellipse centre
            col = colour_img[int(circle[1]), int(circle[0])]

            # Check if it is pink-ish by checking that the Red and Blue values are sufficiently stronger than the Green value
            if (col[2] > RG_ratio * col[1] and col[0] > BG_ratio * col[1]) or True:
                coloured_circles.append(circle)

        # Find our triangle of circles / ellipses by testing every triangle possible from the detected circles / ellipses, and finding the most equilateral triangle
        # Get every combination
        combs = itertools.combinations(coloured_circles, 3)

        # Initialise error as very large
        min_err = 99999999
        # Group is the best triangle found
        group = None

        # Loop through each triangle
        for comb in combs:
            # Compute all the side lengths for the triangle
            side_lengths = [((comb[i][0] - comb[(i + 1) % len(comb)][0])**2 + (comb[i][1] - comb[(i + 1) % len(comb)][1])**2)**0.5 for i in range(len(comb))]

            # Find the mean side length
            mu = np.mean(side_lengths)

            # Reject any triangles with side lengths not within the range, uses SKIP_AMOUNT as the computed side lengths are dependent on the resolution of the image
            if not (min_side_length / SKIP_AMOUNT <= mu <= max_side_length / SKIP_AMOUNT):
                continue

            # Define the error as the sum of the square differences between each side length and the mean side length
            # A perfectly equilateral triangle will have an error of 0 as all side lengths are equal
            err = sum([(s - mu)**2 for s in side_lengths])

            # Check if the error for this triangle is lower than the previous best
            if err < min_err:
                # If so, update the best
                min_err = err
                group = comb

        # Check that the best triangle found is good enough, if not then the circle / ellipse detection probably didn't detect all of our circles / ellipses in this frame of video
        # As sizes / lengths vary with resolution, the square errors vary with resolution squared, so SKIP_AMOUNT**2
        if min_err < error_thresh / SKIP_AMOUNT**2:
            # If so, return True and the given triangle

            # First, sort it so that the biggest circle / ellipse (front) is first
            group = sorted(group, key=lambda x: x[2], reverse=True)
            return True, group
    # If not, return False and no triangle
    return False, None