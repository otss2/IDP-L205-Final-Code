import random
import numpy as np
import socket
import time

class Robot:
    def __init__(self):
        self.position = (0, 0)
        

        self.rotation = 0


        self.prev_xs = []
        self.prev_ys = []
        self.prev_rots = []

        self.LOOKBACK_PERIOD = 10

        self.thresh_x = 4
        self.thresh_y = 4
        self.thresh_angle = 2

        self.last_message_time = time.time()

        self.stop_drive_time = time.time()
        

        self.centre_of_rot = (-0.4, 0) # fraction of side length away from centre in parallel (front is positive) and perp. directions (to the right from its point of view is positive) to rotation

        # SK is a copy of the SKIP_AMOUNT parameter in environ_configs.json
        SK = 2

        # Path set for testing, robot navigates around each point
        self.path_targets = [(600 // SK, 500 // SK), (600 // SK, 60 // SK), (500 // SK, 40 // SK), (400 // SK, 30 // SK), (200 // SK, 40 // SK)]
        self.path_targets = self.path_targets + self.path_targets[::-1]
        self.current_target_index = 0
        self.current_target = self.path_targets[self.current_target_index]
        
        # Calibration values for motor speeds
        self.motor_min_speed = 120
        self.motor_max_speed = 255

        # Proportional constants for feedback loop
        self.k_angle_speed_mode_0 = 4
        self.k_angle_speed_mode_1 = 0


        # Do turning whilst outside outer threshold, until inside inner threshold
        # Then switch to mode 1, until angle error is outside outer threshold
        self.outer_angle_threshold = 10 # degrees either side of straight
        self.inner_angle_threshold = 5
        # Mode 0 is turning on spot, mode 1 is driving forwards whilst slightly adjusting
        self.driving_mode = 0
        
        # Initalise variables
        self.motor_mean_speed = 0
        self.motor_speed_diff = 0

        self.forward_speed = 1 # go at x% of the max speed
        self.turn_speed = 0

        # How different should the motors turn, as one is always slightly faster than the other, 1 means same speed
        self.right_left_motor_speed_ratio = 1

        # Initalise the distance away
        self.distance_away = 1000
        self.time_switched = time.time()

        # Initalise motor speeds
        self.motor_max_speed = self.motor_max_speed / self.right_left_motor_speed_ratio

        self.left_motor_speed = self.motor_mean_speed - self.motor_speed_diff
        self.right_motor_speed = self.motor_mean_speed + self.motor_speed_diff

        self.left_motor_direction = self.left_motor_speed > 0 # False is backwards, True is forwards
        self.right_motor_direction = self.right_motor_speed > 0

        # UDP initalisation
        self.arduino_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.UDP_PORT = 5005
        self.ARDUINO_IP = "192.168.137.215"

        self.arduino_socket.bind(("", self.UDP_PORT))

        self.arduino_socket.settimeout(0.2)


    def update(self, points):
        # point 0 is the big circle, aka front
        # Find mean coord of triangle
        mean_x = np.mean([p[0] for p in points])
        mean_y = np.mean([p[1] for p in points])

        # Find vector that defines the rotation
        angle_vector = (points[0][0] - mean_x, points[0][1] - mean_y)

        angle = np.arctan2(angle_vector[1], angle_vector[0])

        # Find the mean side length
        side_lengths = [((points[i][0] - points[(i + 1) % len(points)][0])**2 + (points[i][1] - points[(i + 1) % len(points)][1])**2)**0.5 for i in range(len(points))]

        mean_side_length = np.mean(side_lengths)

        # Find the centre of rotation in x,y coords
        pos_centre_x = int(mean_x + (self.centre_of_rot[0] * mean_side_length * np.cos(angle)) + (self.centre_of_rot[1] * mean_side_length * -np.sin(angle)))
        pos_centre_y = int(mean_y + (self.centre_of_rot[0] * mean_side_length * np.sin(angle)) + (self.centre_of_rot[1] * mean_side_length * np.cos(angle)))
        
        # Set it as the position
        
        # Outlier detection
        self.prev_xs.append(pos_centre_x)
        self.prev_ys.append(pos_centre_y)
        self.prev_rots.append(angle)

        if len(self.prev_xs) > self.LOOKBACK_PERIOD:
            self.prev_xs.pop(0)
            self.prev_ys.pop(0)
            self.prev_rots.pop(0)

        mu_x = np.mean(self.prev_xs)
        mu_y = np.mean(self.prev_ys)
        mu_angle = np.mean(self.prev_rots)

        stddev_x = np.std(self.prev_xs)
        stddev_y = np.std(self.prev_ys)
        stddev_angle = np.std(self.prev_rots)

        if abs(pos_centre_x - mu_x) > self.thresh_x * stddev_x or abs(pos_centre_y - mu_y) > self.thresh_y * stddev_y or abs(angle - mu_angle) > self.thresh_angle * stddev_angle:
            # Rejecting point
            return

        self.rotation = angle
        self.position = (pos_centre_x, pos_centre_y)

        # Call next update functions
        self.update_path()
        self.update_robot_trajectory()
        self.update_motor_speeds()

    def is_at_position(self, pos, threshold):
        # Returns whether the robot is at a certain position, within a threshold
        # Threshold is e.g. 0.1 for robot being within 10% of pos
        return ( pos[0] * (1 - threshold) <= self.position[0] <= pos[0] * (1 + threshold) ) and ( pos[1] * (1 - threshold) <= self.position[1] <= pos[1] * (1 + threshold) )

    def update_path(self):
        # Decides whether the path needs updating or not
        at_target_threshold = 3 #%
        if self.is_at_position(self.current_target, at_target_threshold / 100):
            print(f"Switching target to {self.current_target}")
            self.current_target_index += 1
            self.current_target_index %= len(self.path_targets)

            self.current_target = self.path_targets[self.current_target_index]

    def update_robot_trajectory(self):
        # Use the new position of the robot, and potentially new target if update_path() does anything, to find the robot's new desired trajectory

        # Find the vector between points, and then calculate the angle from that
        needed_angle_vector = (self.current_target[0] - self.position[0], self.current_target[1] - self.position[1])
        needed_angle = np.arctan2(needed_angle_vector[1], needed_angle_vector[0])

        # Wrap the angle so it is between -pi and +pi, not 0 and +2pi
        needed_angle_change = self.rotation - needed_angle
        if needed_angle_change > np.pi:
            needed_angle_change -= 2 * np.pi

        self.angle_error = needed_angle_change

        # Calculate the distance to the target
        dis_vector = (self.current_target[0] - self.position[0], self.current_target[1] - self.position[1])
        self.distance_away = ((dis_vector[0])**2 + (dis_vector[1])**2)**0.5


    def update_motor_speeds(self):
        # Use the error in the robot's trajectory to find the appropirate motor speeds
        self.motor_speed_diff_mode_1 = self.k_angle_speed_mode_1 * self.angle_error
        self.motor_speed_diff_mode_0 = self.k_angle_speed_mode_0 * self.angle_error

        # Mode detection
        if self.driving_mode == 0:
            if abs(self.angle_error) < self.inner_angle_threshold * (np.pi / 180):
                self.driving_mode = 1
                print("Switching to mode 1")
                self.stop()
                # Stop the robot when switching modes for easy debugging / seeing when it is switching
                self.stop_drive_time = time.time() + 2
                self.time_switched = time.time()
        else:
            if abs(self.angle_error) > self.outer_angle_threshold * (np.pi / 180):
                self.driving_mode = 0
                print("Switching to mode 0")
                self.stop()
                self.stop_drive_time = time.time() + 2
                self.time_switched = time.time()


        if self.driving_mode == 1:
            # Going straight

            # Calculate the required motor speeds
            self.left_motor_speed = self.motor_min_speed + (self.motor_max_speed - self.motor_min_speed) * self.forward_speed
            self.right_motor_speed = self.motor_min_speed + (self.motor_max_speed - self.motor_min_speed) * self.forward_speed
        else:
            # Turning
            # negative angle error means need to turn clockwise - aka left motor forwards, right backwards

            if self.distance_away < 25000 and abs(self.angle_error) * 180 / np.pi < 2 * self.outer_angle_threshold:
                if time.time() - self.time_switched > 0.25: ######
                    self.stop()
                    self.stop_drive_time = time.time() + 0.5
                    self.time_switched = time.time() + 0.5


            motor_speed = (self.motor_min_speed + (self.motor_max_speed - self.motor_min_speed) * self.turn_speed) # e.g. 150

            if self.angle_error < 0:
                self.left_motor_speed = motor_speed
                self.right_motor_speed = -motor_speed
            else:
                self.left_motor_speed = -motor_speed
                self.right_motor_speed = motor_speed

        # Have motor speeds as signed floats, need to convert to absolute ints and also directions

        self.left_motor_direction = self.left_motor_speed > 0 # False is backwards, True is forwards
        self.right_motor_direction = self.right_motor_speed > 0

        self.left_motor_speed = abs(self.left_motor_speed)
        self.right_motor_speed = abs(self.right_motor_speed)

        self.left_motor_speed = int(min(self.left_motor_speed, self.motor_max_speed))
        self.right_motor_speed = int(min(self.right_motor_speed, self.motor_max_speed))

        if self.left_motor_speed % 10 == 0:
            self.left_motor_speed += 1
        if self.right_motor_speed % 10 == 0:
            self.right_motor_speed += 1

        # Motor direction correction
        flip_left_motor = False
        flip_right_motor = False

        if flip_left_motor:
            self.left_motor_direction = not self.left_motor_direction

        if flip_right_motor:
            self.right_motor_direction = not self.right_motor_direction

        # Send these speeds and directions to the Arduino over UDP

        # Create the string
        udp_string = f"{self.left_motor_speed} {int(self.left_motor_direction)} {self.right_motor_speed} {int(self.right_motor_direction)}"

        rand_str = f"  {random.randint(0, 10000)}"
        # Append a random number to the end so for debugging can see which packet is which (often the exact same data so impossible to tell apart)
        udp_string += rand_str

        message = bytes(udp_string, "utf-8")


        # Check a message hasn't been sent too recently
        if time.time() - self.last_message_time > 0.001 and time.time() > self.stop_drive_time:
            # Send the message
            self.arduino_socket.sendto(message, (self.ARDUINO_IP, self.UDP_PORT))
            self.last_message_time = time.time()

    def stop(self):
        udp_string = f"0 {int(self.left_motor_direction)} 0 {int(self.right_motor_direction)}"

        message = bytes(udp_string, "utf-8")
        print("STOPPING")
        print(message)

        self.arduino_socket.sendto(message, (self.ARDUINO_IP, self.UDP_PORT))